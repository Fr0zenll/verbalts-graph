import argparse
import os
import random
import shutil
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cttp.cttp_model import CTTP


DEFAULT_CONFIG = {
    "clip_type": "clip_patchtst",
    "device": "cuda:0",
    "loss_type": "CE",
    "text": {
        "coemb_dim": 512,
        "device": "cuda:0",
        "llm_finetune": "frozen",
        "pretrain_model_dim": 768,
        "pretrain_model_path": "/data/kangjiale/verbalts-graph/VerbalTS-main/save/Longclip",
        "textemb_hidden_dim": 1024,
        "output_type": "all",
    },
    "ts": {
        "activation": "gelu",
        "coemb_dim": 512,
        "d_ff": 256,
        "d_model": 128,
        "device": "cuda:0",
        "dropout": 0.1,
        "e_layers": 2,
        "factor": 1,
        "n_heads": 8,
        "n_var": 10,
        "output_attention": True,
        "padding": 8,
        "patch_len": 16,
        "pretrain_encoder_path": "",
        "seq_len": 240,
        "stride": 8,
        "type": "patchtst_mae_pretrain",
    },
}


class SMDNLCTTPDataset(Dataset):
    def __init__(self, data_dir, split="train", text_variant="random"):
        self.ts_data = np.load(os.path.join(data_dir, f"{split}_ts.npy"))
        self.text_data = np.load(os.path.join(data_dir, f"{split}_text_caps.npy"), allow_pickle=True)
        self.split = split
        self.text_variant = text_variant

        if self.ts_data.ndim != 3:
            raise ValueError(f"{split}_ts.npy must be 3D [N, L, V], got {self.ts_data.shape}.")
        if self.text_data.ndim != 2:
            raise ValueError(
                f"{split}_text_caps.npy must be 2D [N, variants], got {self.text_data.shape}."
            )
        if len(self.ts_data) != len(self.text_data):
            raise ValueError(
                f"{split} ts/text sample count mismatch: {len(self.ts_data)} vs {len(self.text_data)}."
            )

    def __len__(self):
        return len(self.ts_data)

    def _select_text(self, idx):
        variants = self.text_data[idx]
        if self.text_variant == "random":
            return str(random.choice(variants))
        variant_idx = int(self.text_variant)
        return str(variants[variant_idx])

    def __getitem__(self, idx):
        # Official CTTP expects [L, V]. Do not transpose to [V, L].
        ts = torch.tensor(self.ts_data[idx], dtype=torch.float32)
        text = self._select_text(idx)
        ts_len = torch.tensor(ts.shape[0], dtype=torch.int)
        return ts, ts_len, text


def collate_fn(batch):
    ts, ts_len, text = zip(*batch)
    return torch.stack(ts, dim=0), torch.stack(ts_len, dim=0), list(text)


def load_or_create_config(args):
    if args.config_path:
        with open(args.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = DEFAULT_CONFIG

    config["device"] = args.device
    config["text"]["device"] = args.device
    config["ts"]["device"] = args.device
    config["text"]["pretrain_model_path"] = args.longclip_path
    config["ts"]["seq_len"] = args.seq_len
    config["ts"]["n_var"] = args.n_vars
    return config


def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for ts, ts_len, text in loader:
            ts = ts.to(device)
            ts_len = ts_len.to(device)
            loss = model(ts, ts_len, text, None)["all"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation CTTP loss: {loss.item()}")
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    config = load_or_create_config(args)
    config["device"] = str(device)
    config["text"]["device"] = str(device)
    config["ts"]["device"] = str(device)

    saved_config_path = os.path.join(args.save_dir, "model_configs.yaml")
    with open(saved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    train_dataset = SMDNLCTTPDataset(
        args.data_dir,
        split="train",
        text_variant=args.text_variant,
    )
    valid_dataset = SMDNLCTTPDataset(
        args.data_dir,
        split="valid",
        text_variant=args.valid_text_variant,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print("Saved CTTP config to:", saved_config_path)
    print("Train samples:", len(train_dataset))
    print("Valid samples:", len(valid_dataset))
    print("Config seq_len:", config["ts"]["seq_len"], "n_var:", config["ts"]["n_var"])

    model = CTTP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (ts, ts_len, text) in enumerate(train_loader):
            ts = ts.to(device)
            ts_len = ts_len.to(device)

            optimizer.zero_grad()
            loss_dict = model(ts, ts_len, text, None)
            loss = loss_dict["all"]
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite CTTP loss at epoch {epoch}, batch {batch_idx}: {loss.item()}"
                )
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.gradient_clip,
                    error_if_nonfinite=True,
                )
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        train_loss = total_loss / max(len(train_loader), 1)
        valid_loss = evaluate_loss(model, valid_loader, device)
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_path = os.path.join(args.save_dir, "clip_model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Best CTTP updated: {best_valid_loss:.6f}. Saved to {best_path}")

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            epoch_path = os.path.join(args.save_dir, f"cttp_epoch_{epoch}.pth")
            torch.save(model.state_dict(), epoch_path)
            print("Saved epoch checkpoint to:", epoch_path)

    final_path = os.path.join(args.save_dir, "cttp_final.pth")
    torch.save(model.state_dict(), final_path)
    if not os.path.exists(os.path.join(args.save_dir, "clip_model_best.pth")):
        shutil.copyfile(final_path, os.path.join(args.save_dir, "clip_model_best.pth"))
    print("CTTP training done. Final checkpoint:", final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/kangjiale/verbalts-graph/data/smd_nl")
    parser.add_argument("--save_dir", type=str, default="./save/SMD_NL_cttp")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument(
        "--longclip_path",
        type=str,
        default="/data/kangjiale/verbalts-graph/VerbalTS-main/save/Longclip",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_vars", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=240)
    parser.add_argument("--text_variant", type=str, default="random")
    parser.add_argument("--valid_text_variant", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=10)

    main(parser.parse_args())
