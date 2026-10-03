import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CustomRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(norm_x + self.eps))

class CustomSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)
        self.w3.IS_RESIDUAL_PROJECTION = True

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# --- Mixture of Experts with Shared Expert (DeepSeek-MoE style) ---
class MoELayer(nn.Module):
    """
    Shared Expert + Routed Experts.
    
    The shared expert fires for EVERY token (captures universal patterns like grammar).
    The router additionally picks top_k specialized experts per token.
    This prevents expert collapse far better than load-balancing loss alone.
    """
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        expert_hidden = dim * 2
        
        # Shared expert: always active
        self.shared_expert = CustomSwiGLU(dim, expert_hidden)
        
        # Routed experts: router selects top_k per token
        self.experts = nn.ModuleList([CustomSwiGLU(dim, expert_hidden) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        self.aux_loss = 0.0

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        # Shared expert output (always computed for every token)
        shared_out = self.shared_expert(x_flat)
        
        # Router scores for the specialized experts
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Load-balancing auxiliary loss
        one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts).float().sum(dim=1)
        f = one_hot.mean(dim=0)
        p = router_probs.mean(dim=0)
        self.aux_loss = self.num_experts * (f * p).sum()
        
        # Scatter/gather expert dispatch — one loop, one batched forward per expert
        routed_out = torch.zeros_like(x_flat)
        flat_expert_indices = top_k_indices.view(-1)
        flat_weights = top_k_weights.view(-1)
        flat_token_indices = torch.arange(x_flat.size(0), device=x_flat.device) \
            .unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        for e_idx in range(self.num_experts):
            mask = (flat_expert_indices == e_idx)
            if mask.any():
                token_ids = flat_token_indices[mask]
                expert_out = self.experts[e_idx](x_flat[token_ids])
                routed_out.index_add_(0, token_ids, 
                                      flat_weights[mask].unsqueeze(-1) * expert_out)
        
        # Combine: shared + routed
        output = shared_out + routed_out
        return output.view(B, T, C)

# --- Rotary Positional Embeddings (RoPE) ---
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# --- Grouped Query Attention with QK-Norm and Sliding Window ---
class CustomGroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len, dropout=0.0, window_size=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj.IS_RESIDUAL_PROJECTION = True
        
        self.q_norm = CustomRMSNorm(self.head_dim)
        self.k_norm = CustomRMSNorm(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Build causal mask — either full or sliding window
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        if window_size is not None:
            # Zero out positions beyond the window: token i can only see [i-window+1, i]
            mask = mask - torch.tril(torch.ones(max_seq_len, max_seq_len), diagonal=-(window_size + 1))
        self.register_buffer('mask', mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x, freqs_cis, kv_cache=None):
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis[:T])
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # KV Cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            if cached_k is not None:
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
                # Sliding window: truncate cache to window_size
                if self.window_size is not None and k.size(2) > self.window_size:
                    k = k[:, :, -self.window_size:]
                    v = v[:, :, -self.window_size:]
            new_cache = (k, v)
            T_full = k.size(2)
        else:
            # If no cache is passed, this is the first prompt evaluation.
            # We must return the current k, v to start the cache!
            new_cache = (k, v)
            T_full = T
        
        # GQA replication
        if self.n_rep > 1:
            k_exp = k.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T_full, self.head_dim).reshape(B, self.n_heads, T_full, self.head_dim)
            v_exp = v.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T_full, self.head_dim).reshape(B, self.n_heads, T_full, self.head_dim)
        else:
            k_exp, v_exp = k, v

        att = (q @ k_exp.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Gemma 2: Attention Logit Soft-Capping
        att = 50.0 * torch.tanh(att / 50.0)
        
        if kv_cache is None:
            att = att.masked_fill(self.mask[:, :, :T, :T_full] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = (att @ v_exp).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y)), new_cache

class CustomBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len, num_experts, top_k, dropout, window_size=None):
        super().__init__()
        self.norm1 = CustomRMSNorm(dim)
        self.attn = CustomGroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len, dropout, window_size)
        self.norm2 = CustomRMSNorm(dim)
        self.ffn = MoELayer(dim, num_experts=num_experts, top_k=top_k)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        attn_out, _ = self.attn(self.norm1(x), freqs_cis, kv_cache=None)
        x = x + attn_out
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def forward_with_cache(self, x, freqs_cis, kv_cache):
        attn_out, new_cache = self.attn(self.norm1(x), freqs_cis, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache

class MyCustomLLM(nn.Module):
    def __init__(self, vocab_size, dim=512, n_layers=8, n_heads=8, n_kv_heads=2, 
                 max_seq_len=512, num_experts=4, top_k=2, dropout=0.1,
                 window_size=128, n_predict=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_predict = n_predict
        self.embed_scale = math.sqrt(dim)
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        # Alternating attention windows: even layers = full, odd layers = sliding window
        self.layers = nn.ModuleList([
            CustomBlock(
                dim, n_heads, n_kv_heads, max_seq_len, num_experts, top_k, dropout,
                window_size=(window_size if i % 2 == 1 else None)
            )
            for i in range(n_layers)
        ])
        self.norm = CustomRMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Multi-token prediction: extra heads that predict tokens +2, +3, etc.
        # These force the hidden representations to encode richer future context.
        # Discarded at inference — zero cost to generation speed.
        if n_predict > 1:
            self.extra_heads = nn.ModuleList([
                nn.Linear(dim, vocab_size, bias=False) for _ in range(n_predict - 1)
            ])
        
        # Weight tying (main head only)
        self.embed.weight = self.output.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'IS_RESIDUAL_PROJECTION'):
                module.weight.data.normal_(mean=0.0, std=(0.02 / math.sqrt(2 * self.n_layers)))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _collect_aux_loss(self):
        aux = 0.0
        for layer in self.layers:
            aux += layer.ffn.aux_loss
        return aux / self.n_layers

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.embed(idx) * self.embed_scale
        x = self.embed_dropout(x)
        
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
            
        x = self.norm(x)
        
        if targets is not None:
            # Main head: predict next token (position +1)
            logits = self.output(x)
            logits = 30.0 * torch.tanh(logits / 30.0)  # Gemma 2: Final Logit Soft-Capping
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Multi-token prediction: extra heads predict +2, +3, ...
            if self.n_predict > 1:
                for k, head in enumerate(self.extra_heads, start=2):
                    extra_logits = head(x)
                    extra_logits = 30.0 * torch.tanh(extra_logits / 30.0) # Cap extra heads too
                    # Build shifted targets: position i should predict token at i+k
                    # targets[i] = token[i+1], so we want targets[i+k-1]
                    shifted = torch.full_like(targets, -1)  # -1 = ignore
                    if k - 1 < T:
                        shifted[:, :T-k+1] = targets[:, k-1:]
                    extra_loss = F.cross_entropy(
                        extra_logits.view(-1, extra_logits.size(-1)),
                        shifted.view(-1), ignore_index=-1
                    )
                    loss = loss + extra_loss
                loss = loss / self.n_predict
            
            # MoE auxiliary loss
            loss = loss + 0.01 * self._collect_aux_loss()
        else:
            logits = self.output(x[:, [-1], :]) 
            logits = 30.0 * torch.tanh(logits / 30.0)
            loss = None
            
        return logits, loss

    def forward_with_cache(self, idx, kv_cache=None):
        """Inference with KV cache. Only uses the main prediction head."""
        B, T = idx.size()
        start_pos = 0 if kv_cache is None else kv_cache[0][0].size(2)
        
        x = self.embed(idx) * self.embed_scale
        freqs_slice = self.freqs_cis[start_pos : start_pos + T]
        
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = layer.forward_with_cache(x, freqs_slice, layer_cache)
            new_kv_cache.append(new_cache)
            
        x = self.norm(x)
        logits = self.output(x)
        logits = 30.0 * torch.tanh(logits / 30.0)
        return logits, new_kv_cache
