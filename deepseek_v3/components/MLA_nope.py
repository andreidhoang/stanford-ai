
import torch
import torch.nn as nn
import torch.nn.functional as F


class RopelessMLA(nn.Module):
    def __init__(self, d_model, n_heads, kv_latent_dim):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.dh = d_model // n_heads

        # projection layers
        self.W_q = nn.Linear(d_model, d_model, bias=False) # query projection
        self.W_dkv = nn.Linear(d_model, kv_latent_dim, bias=False) # compress into latent KV space
        self.W_uk = nn.Linear(kv_latent_dim, d_model, bias=False) # decompress K
        self.W_uv = nn.Linear(kv_latent_dim, d_model, bias=False) # decompress V
        self.W_o = nn.Linear(d_model, d_model, bias=False) # final output projection

        self.ln = nn.LayerNorm(kv_latent_dim)

    def forward(self, x, kv_cache=None, past_length=0):
        B, S, D = x.size()

        # basic sanity checks
        assert D == self.d_model, f"input feature dim (={D}) does not match d_model (={self.d_model})"

        # project queries with W_q (we compute this once per forward)
        # note: earlier version attempted to "absorb" W_q into W_uk but the
        # shapes/ordering made the math incorrect unless W_q is block-diagonal.
        q_proj = self.W_q(x)  # (B, S, D)
        # compress x into latent KV space
        new_c_kv = self.ln(self.W_dkv(x)) # (B, S, latent_dim)
        if kv_cache is None:
            c_kv = new_c_kv
        else:
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1) # (B, S_total, latent_dim)
        S_full = c_kv.size(1)

        # decompress V to full d_model and split into heads

        v_full = self.W_uv(c_kv) # (B, S_full, D)
        v = v_full.view(B, S_full, self.n_heads, self.dh).transpose(1,2) # (B, n_heads, S_full, dh)

        # split into heads: shape -> (B, n_heads, S, dh)
        q = q_proj.view(B, S, self.n_heads, self.dh).transpose(1, 2)

        # prepare per-head K weights (decompress weights): (n_heads, dh, latent_dim)
        k_weights = self.W_uk.weight.view(self.n_heads, self.dh, -1)

        # compute attn scores (per-head)
        attn_scores = torch.zeros(B, self.n_heads, S, S_full, device=x.device)  # (B, n_heads, S, S_full)
        for h in range(self.n_heads):
            # tmp: (B, S, latent_dim)
            tmp = torch.matmul(q[:, h], k_weights[h])
            attn_scores[:, h] = torch.bmm(tmp, c_kv.transpose(1, 2))  # (B, S, S_full)

        # scale and apply causal mask
        attn_scores = attn_scores / (self.dh ** 0.5)
        mask = torch.tril(torch.ones((S, S_full), device=x.device), diagonal=past_length)
        attn_scores = attn_scores.masked_fill(mask.view(1,1,S,S_full)==0, float('-inf'))

        # softmax to get attn weights
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, n_heads, S, S_full) 

        # apply attn weights to each head's V separately
        out_heads = []
        for h in range(self.n_heads):
            # (B, S, dh)
            context_h = torch.bmm(attn_weights[:, h], v[:, h])
            out_heads.append(context_h)

        # combine heads -> (B, S, D)
        out = torch.stack(out_heads, dim=2)  # (B, S, n_heads, dh)
        out = out.contiguous().view(B, S, self.d_model)
        out = self.W_o(out)

        # return output and updated kv cache (so caller can pass it next step)
        return out, c_kv
        
