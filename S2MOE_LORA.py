import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ======================
# 3. Define LoRA & MoE
# ======================

class LoRALayer(nn.Module):
    def __init__(self, fan_in, fan_out, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros((rank, fan_in)))
        self.lora_B = nn.Parameter(torch.zeros((fan_out, rank)))
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, X):
        return (self.lora_dropout(X) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.shape
        return cls(fan_in, fan_out, rank, lora_dropout_p, lora_alpha)

class LoRA_MOE_LM(nn.Module):
    def __init__(self, original_module, num_experts=4, rank=8, alpha=32, dense_moe=False):
        super().__init__()
        self.num_experts = num_experts
        self.rank = rank
        self.alpha = alpha
        self.dense_moe = dense_moe
        # Store original_module as a non-registered attribute to avoid circular module references
        object.__setattr__(self, 'original_module_', original_module)
        self._aux_losses = {}
        self._last_gate_value = None

        d_model = original_module.gate_proj.in_features
        mlp_width = original_module.gate_proj.out_features

        self.moe_gate = nn.ModuleList()
        self.moe_up = nn.ModuleList()
        self.moe_down = nn.ModuleList()

        for _ in range(num_experts):
            self.moe_gate.append(LoRALayer.from_linear(nn.Linear(d_model, mlp_width), rank, 0.05, alpha))
            self.moe_up.append(LoRALayer.from_linear(nn.Linear(d_model, mlp_width), rank, 0.05, alpha))
            self.moe_down.append(LoRALayer.from_linear(nn.Linear(mlp_width, d_model), rank, 0.05, alpha))

        self.router = nn.Linear(d_model, num_experts)

        # Freeze original MLP
        for n, p in self.original_module_.named_parameters():
            p.requires_grad = False

    def forward_lora_moe(self, x, original_proj, routing, moe):
        original_out = original_proj(x)
        lora_out = torch.stack([m(x) for m in moe], dim=2)
        lora_out = (lora_out * routing[..., None]).sum(2)
        return original_out + lora_out

    def forward(self, x):
        logits = self.router(x)
        routing = F.softmax(logits, dim=-1)
        top_expert = routing.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, top_expert, 1.0)
        routing_final = y_hard - routing.detach() + routing

        self._last_gate_value = routing.detach()

        # simple load balancing loss
        if self.training:
            # Encourage uniform expert usage
            mean_routing = routing.mean(dim=[0, 1])  # [num_experts]
            uniform = torch.ones_like(mean_routing) / self.num_experts
            # KL divergence from uniform
            Lb = (mean_routing * (mean_routing.clamp_min(1e-10).log() - uniform.log())).sum()
            self._aux_losses = {'Lb': Lb}
        else:
            self._aux_losses = {}

        gate_out = self.forward_lora_moe(x, self.original_module_.gate_proj, routing_final, self.moe_gate)
        up_out = self.forward_lora_moe(x, self.original_module_.up_proj, routing_final, self.moe_up)
        x = self.original_module_.act_fn(gate_out) * up_out
        x = self.forward_lora_moe(x, self.original_module_.down_proj, routing_final, self.moe_down)
        return x
    
class S2MoE_LoRA_MLP(nn.Module):
    """
    S2MoE + LoRA MLP:
      - Multi-expert LoRA
      - Learnable noise parameters (α, β)
      - InfoNCE + Load balancing auxiliary losses
    """

    def __init__(
        self,
        original_module,
        num_experts=4,
        rank=8,
        alpha=32,
        lora_dropout=0.05,
        top_k=1,
        alpha_bal=0.01,   # weight for load balancing loss
        beta_unc=0.1,     # weight for InfoNCE loss
    ):
        super().__init__()

        self.num_experts = num_experts
        self.rank = rank
        self.alpha = alpha
        self.top_k = top_k
        self.alpha_bal = alpha_bal
        self.beta_unc = beta_unc

        # Store original_module as a non-registered attribute to avoid circular module references
        # We use object.__setattr__ to bypass PyTorch's module registration
        object.__setattr__(self, 'original_', original_module)
        d_model = original_module.gate_proj.in_features
        mlp_width = original_module.gate_proj.out_features

        # Expert LoRA layers
        self.moe_gate = nn.ModuleList()
        self.moe_up = nn.ModuleList()
        self.moe_down = nn.ModuleList()
        for _ in range(num_experts):
            self.moe_gate.append(LoRALayer.from_linear(nn.Linear(d_model, mlp_width), rank, lora_dropout, alpha))
            self.moe_up.append(LoRALayer.from_linear(nn.Linear(d_model, mlp_width), rank, lora_dropout, alpha))
            self.moe_down.append(LoRALayer.from_linear(nn.Linear(mlp_width, d_model), rank, lora_dropout, alpha))

        # Router + merge gate
        self.router = nn.Linear(d_model, num_experts)
        self.merge_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        # Learnable noise scales (α, β)
        self.alpha_noise = nn.Parameter(torch.ones(1, 1, d_model))
        self.beta_noise = nn.Parameter(torch.zeros(1, 1, d_model))

        # Freeze original MLP parameters
        for p in self.original_.parameters():
            p.requires_grad = False

        self._aux_losses = {}

    # ========================
    # Utility Functions
    # ========================
    @staticmethod
    def _topk_ste(probs, k=1):
        """Straight-through estimator for top-k routing"""
        topk_idx = probs.topk(k, dim=-1).indices
        y_hard = torch.zeros_like(probs).scatter_(-1, topk_idx, 1.0)
        return y_hard + (probs - probs.detach())

    def _compute_load_balancing(self, probs):
        """Encourage balanced expert usage"""
        p = probs.reshape(-1, probs.size(-1))
        mean_p = p.mean(dim=0)
        E = p.size(-1)
        uniform = torch.full_like(mean_p, 1.0 / E)
        kl = (mean_p * (mean_p.clamp_min(1e-8).log() - uniform.log())).sum()
        return kl

    def _info_nce(self, x, x_hat):
        """InfoNCE loss between x and its noisy version"""
        x1 = F.normalize(x.reshape(-1, x.size(-1)), dim=-1)
        x2 = F.normalize(x_hat.reshape(-1, x_hat.size(-1)), dim=-1)
        logits = x1 @ x2.t()
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)
    
    # Gaussian Noise Module (per batch)
    def add_gaussian_noise(self, x):
        """
        Implements: x̂ = N(1, σ_x^2)·x + N(μ_x, σ_x^2)
        where μ_x, σ_x are computed over feature dimensions per batch.
        """
        # Compute mean and std across token dimension (per sample)
        std, mean = torch.std_mean(x, dim=1, keepdim=True)

        self._last_noise_std = std.detach()
        self._last_noise_mean = mean.detach()

        # Sample N1, N2 ~ Gaussian using batch statistics
        N1 = torch.randn_like(x) * std + 1.0      # N(1, σ_x^2)
        N2 = torch.randn_like(x) * std + mean     # N(μ_x, σ_x^2)

        # Apply batch-dependent noise
        x_hat = N1 * x + N2
        return x_hat

    def _lora_moe_proj(self, x, original_proj, routing, experts):
        base = original_proj(x) # [B, L, d_out]
        #exp_out = torch.stack([m(x) for m in experts], dim=-1)  # [..., d_out, E]

        # debug cuda out of memory
        B, L, d_out = base.shape
        E = len(experts)
        exp_out = torch.zeros(B, L, d_out, E, device=x.device, dtype=x.dtype)
        for i, expert in enumerate(experts):
            exp_out[:, :, :, i] = expert(x)
        exp_mix = (exp_out * routing.unsqueeze(-2)).sum(dim=-1)  # sum over experts
        return base + exp_mix

    def forward_once(self, x, routing):
        gate = self._lora_moe_proj(x, self.original_.gate_proj, routing, self.moe_gate)
        up = self._lora_moe_proj(x, self.original_.up_proj, routing, self.moe_up)
        h = self.original_.act_fn(gate) * up
        return self._lora_moe_proj(h, self.original_.down_proj, routing, self.moe_down)

    # ========================
    # Forward Pass
    # ========================
    def forward(self, x):
        self._aux_losses = {}

        # routing
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        routing = self._topk_ste(probs, k=self.top_k)

        # gating probs
        self._last_gating_probs = probs.detach()

        # Load balancing loss
        Lb = self._compute_load_balancing(probs)
        self._aux_losses['Lb'] = Lb

        # Clean path
        y_clean = self.forward_once(x, routing)

        # Noisy path (only during training)
        if self.training and self.beta_unc > 0:
            # noise = torch.randn_like(x)
            # x_hat = self.alpha_noise * x + self.beta_noise * noise
            x_hat = self.add_gaussian_noise(x)
            y_noisy = self.forward_once(x_hat, routing)
            Lu = self._info_nce(x, x_hat)
            self._aux_losses['Lu'] = Lu
        else:
            y_noisy = torch.zeros_like(y_clean)
            self._aux_losses['Lu'] = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Merge gate
        g = self.merge_gate(x)

        self._last_gate_value = g.detach()

        y = g * y_clean + (1 - g) * y_noisy
        return y


class MlpWithLoRAMoE(nn.Module):
    """Wrapper that replaces a layer's MLP with a LoRA-MoE-enabled module.

    This avoids creating circular references by not registering the base MLP
    as a submodule, only the LoRA-MoE layer which contains LoRA adapters and gates.
    This ensures compatibility with HuggingFace when pushing models to the hub.
    """

    def __init__(self, base_mlp: nn.Module, lora_moe_layer: LoRA_MOE_LM) -> None:
        super().__init__()
        # Don't register base_mlp as a submodule to avoid circular reference
        # It's already stored in lora_moe_layer.original_module_
        self.lora_moe_layer = lora_moe_layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Delegate to the LoRA-MoE forward
        return self.lora_moe_layer(hidden_states)


class MlpWithS2MoELoRA(nn.Module):
    """Wrapper that replaces a layer's MLP with a S2MoE-LoRA-enabled module.

    This avoids creating circular references by not registering the base MLP
    as a submodule, only the S2MoE-LoRA layer which contains LoRA adapters and gates.
    This ensures compatibility with HuggingFace when pushing models to the hub.
    """

    def __init__(self, base_mlp: nn.Module, s2moe_layer: S2MoE_LoRA_MLP) -> None:
        super().__init__()
        # Don't register base_mlp as a submodule to avoid circular reference
        # It's already stored in s2moe_layer.original_
        self.s2moe_layer = s2moe_layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Delegate to the S2MoE-LoRA forward
        return self.s2moe_layer(hidden_states)