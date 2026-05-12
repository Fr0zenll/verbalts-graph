import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Basic graph convolution layer.

    Args:
        x: Tensor with shape (..., V, C_in), where V is the number of variables.
        adj: Tensor with shape (V, V). It can be a torch.Tensor converted from npy.

    Returns:
        Tensor with shape (..., V, C_out).
    """
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        if x.ndim < 2:
            raise ValueError(f"x must have at least 2 dims (..., V, C), got {x.shape}.")
        if adj.ndim != 2:
            raise ValueError(f"adj must be 2D with shape (V, V), got {adj.shape}.")
        if adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj must be square, got {adj.shape}.")
        if x.shape[-2] != adj.shape[0]:
            raise ValueError(
                f"x variable dim and adj size mismatch: x={x.shape}, adj={adj.shape}."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"x feature dim must be {self.input_dim}, got {x.shape[-1]}."
            )

        adj = adj.to(device=x.device, dtype=x.dtype)
        support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class SimpleVariableGCN(nn.Module):
    """
    GCN for message passing over the variable dimension.

    Supported inputs:
        1. x: (B, T, V)
           Each variable node has one scalar value per time step. The module
           temporarily uses feature dim C=1 and returns (B, T, V).

        2. x: (B, T, V, C)
           Each variable node has C hidden features per time step. The module
           returns (B, T, V, C) when output_dim == C.
    """
    def __init__(
        self,
        num_variables,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        num_layers=1,
        dropout=0.0,
        residual=True,
        activation="relu",
    ):
        super().__init__()
        if num_variables <= 0:
            raise ValueError(f"num_variables must be positive, got {num_variables}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.num_variables = num_variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.activation_name = activation

        dims = [input_dim]
        if num_layers == 1:
            dims.append(output_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(output_dim)

        self.layers = nn.ModuleList(
            [GraphConvolution(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def _activate(self, x):
        if self.activation_name == "relu":
            return F.relu(x)
        if self.activation_name == "gelu":
            return F.gelu(x)
        if self.activation_name == "silu":
            return F.silu(x)
        if self.activation_name in ("none", None):
            return x
        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def forward(self, x, adj):
        """
        Args:
            x:
                3D tensor with shape (B, T, V), or
                4D tensor with shape (B, T, V, C).
            adj:
                Graph adjacency with shape (V, V). Usually this is the
                GCN-normalized adjacency matrix saved as graph_adj.npy.

        Returns:
            If input is (B, T, V), returns (B, T, V).
            If input is (B, T, V, C), returns (B, T, V, output_dim).
            For shape-preserving use with 4D tensors, set output_dim == C.
        """
        if x.ndim not in (3, 4):
            raise ValueError(f"x must be 3D or 4D, got shape {x.shape}.")
        if adj.ndim != 2:
            raise ValueError(f"adj must be 2D with shape (V, V), got {adj.shape}.")
        if adj.shape != (self.num_variables, self.num_variables):
            raise ValueError(
                f"adj shape must be ({self.num_variables}, {self.num_variables}), "
                f"got {adj.shape}."
            )

        is_3d = x.ndim == 3
        if is_3d:
            B, T, V = x.shape
            if V != self.num_variables:
                raise ValueError(
                    f"x variable dim must be {self.num_variables}, got {V}."
                )
            if self.input_dim != 1:
                raise ValueError(
                    f"3D input uses scalar node features, so input_dim must be 1, "
                    f"got {self.input_dim}."
                )
            h = x.unsqueeze(-1)  # (B, T, V) -> (B, T, V, 1)
        else:
            B, T, V, C = x.shape
            if V != self.num_variables:
                raise ValueError(
                    f"x variable dim must be {self.num_variables}, got {V}."
                )
            if C != self.input_dim:
                raise ValueError(
                    f"x feature dim must be input_dim={self.input_dim}, got {C}."
                )
            h = x

        residual_input = h
        adj = adj.to(device=h.device, dtype=h.dtype)

        for layer_id, layer in enumerate(self.layers):
            h = layer(h, adj)
            if layer_id < len(self.layers) - 1:
                h = self._activate(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self.residual and h.shape == residual_input.shape:
            h = h + residual_input

        if is_3d:
            if h.shape[-1] != 1:
                raise ValueError(
                    f"3D input requires output_dim=1 to return (B, T, V), "
                    f"got output feature dim {h.shape[-1]}."
                )
            return h.squeeze(-1)
        return h


VariableGCN = SimpleVariableGCN
