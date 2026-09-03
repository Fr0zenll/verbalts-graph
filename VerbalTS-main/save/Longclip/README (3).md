---
datasets:
- SPRIGHT-T2I/spright_coco
base_model: BeichenZhang/LongCLIP-L
---

# 🔥 Update SUMMER 2025: 🔥
🤖 New and greatly improved version of the model, check out:
- 🌑 [https://huggingface.co/zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14)

----
## A fine-tune of Long-CLIP - original model: [BeichenZhang/LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L)
- ❤️ this CLIP? [Help feed it](https://ko-fi.com/zer0int) if you can. Besides data, CLIP eats time & expensive electricity of DE. TY! 🤗
- Want to feed it yourself? All code for fine-tuning and much more is on [my GitHub](https://github.com/zer0int).
----
- # Note for using Long-CLIP as the Text Encoder with Flux.1, SDXL, Stable Diffusion: 
- Get the ComfyUI Long-CLIP nodes here: [https://github.com/SeaArtLab/ComfyUI-Long-CLIP](https://github.com/SeaArtLab/ComfyUI-Long-CLIP)
- If you don't use Comfy, it's at least a starting point for reverse engineering & applying it to your code! 🤗
----
# 🚨 IMPORTANT NOTE for loading with HuggingFace Transformers: 👀

```
model_id = "zer0int/LongCLIP-GmP-ViT-L-14"

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
```
# ❌ Error due to mismatch with defined 77 tokens in Transformers library

# 👇
# Option 1 (simple & worse):
Truncate to 77 tokens
`CLIPModel.from_pretrained(model_id, ignore_mismatched_sizes=True)`

```
# Cosine similarities for 77 tokens is WORSE:
# tensor[photo of a cat, picture of a dog, cat, dog] # image ground truth: cat photo
tensor([[0.16484, 0.0749, 0.1618, 0.0774]], device='cuda:0') 📉
```
# 👇
# Option 2, proper integration: 💖 RECOMMENDED 💖

- ### Solution for implementation of 248 tokens / thanks [@kk3dmax ](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14/discussions/3) 🤗
- Obtain a full example script using this solution for Flux.1 inference on [my GitHub](https://github.com/zer0int/CLIP-txt2img-diffusers-scripts)

```
model_id = ("zer0int/LongCLIP-GmP-ViT-L-14")
config = CLIPConfig.from_pretrained(model_id)
config.text_config.max_position_embeddings = 248
clip_model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype, config=config)
clip_processor = CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=248)

pipe.tokenizer = clip_processor.tokenizer  # Replace with the CLIP tokenizer
pipe.text_encoder = clip_model.text_model  # Replace with the CLIP text encoder
pipe.tokenizer_max_length = 248
pipe.text_encoder.dtype = torch.bfloat16
```

```
# Resulting Cosine Similarities for 248 tokens padded:
# tensor[photo of a cat, picture of a dog, cat, dog] -- image ground truth: cat photo
tensor([[0.2128, 0.0978, 0.1957, 0.1133]], device='cuda:0') ✅
```

----
## Update 12/AUG/2024:
New *BEST* model, custom loss with label smoothing.
Small gain for a diverse and large good quality dataset, but big relative gains for an overfit-prone fine-tune (small batch size, 1 GPU, narrow dataset of e.g. 'sneakers', etc.) are possible!
Fine-tune your model with the provided code for GmP-Smooth: [https://github.com/zer0int/Long-CLIP](https://github.com/zer0int/Long-CLIP)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6490359a877fc29cb1b09451/l3FYkaicihqXv5D9wLDAF.png)

----

The fine-tune has an improved ImageNet/ObjectNet accuracy of 0.89 (original Long-CLIP by the authors:~0.81)**.


Made possible with Geometric Parametrization (GmP):

```

"Normal" CLIP MLP (multi-layer perceptron):

(mlp): Sequential(
  |-(c_fc): Linear(in_features=1024, out_features=4096, bias=True)
  | (gelu): QuickGELU()
|-}-(c_proj): Linear(in_features=4096, out_features=1024, bias=True)
| | 
| |-- visual.transformer.resblocks.0.mlp.c_fc.weight
| |-- visual.transformer.resblocks.0.mlp.c_fc.bias
|
|---- visual.transformer.resblocks.0.mlp.c_proj.weight
|---- visual.transformer.resblocks.0.mlp.c_proj.bias


GmP CLIP MLP:

Weight decomposition into:
- radial component 'r' as norm of pre-trained weights
- angular component 'theta' as normalized direction
-> preserves weight vectors' directionality and magnitude

(mlp): Sequential(
  |-(c_fc): GeometricLinear()
  | (gelu): QuickGELU()
|-}-(c_proj): GeometricLinear()
| | 
| |-- visual.transformer.resblocks.0.mlp.c_fc.r
| |-- visual.transformer.resblocks.0.mlp.c_fc.theta
| |-- visual.transformer.resblocks.0.mlp.c_fc.bias
|
|---- visual.transformer.resblocks.0.mlp.c_proj.r
|---- visual.transformer.resblocks.0.mlp.c_proj.theta
|---- visual.transformer.resblocks.0.mlp.c_proj.bias

(Same thing for [text] transformer.resblocks)

```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6490359a877fc29cb1b09451/OqhNxW-D9c58mkZyUQlL_.png)

✅ The model / state_dict I am sharing was converted back to .weight after fine-tuning - alas, it can be used in the same manner as any state_dict, e.g. for use with ComfyUI as the SDXL / SD3 Text Encoder using [SeaArtLab/ComfyUI-Long-CLIP](https://github.com/SeaArtLab/ComfyUI-Long-CLIP) custom nodes! 🤗

** For details on training and those numbers / the eval, or for just fine-tuning the model yourself, see: [https://github.com/zer0int/Long-CLIP](https://github.com/zer0int/Long-CLIP)

```
@article{zhang2024longclip,
        title={Long-CLIP: Unlocking the Long-Text Capability of CLIP},
        author={Beichen Zhang and Pan Zhang and Xiaoyi Dong and Yuhang Zang and Jiaqi Wang},
        journal={arXiv preprint arXiv:2403.15378},
        year={2024}
}
```

Pre-trained CLIP model by OpenAI, License: [MIT License](https://github.com/openai/CLIP/blob/main/LICENSE)