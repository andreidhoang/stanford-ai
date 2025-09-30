import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import requests
torch.manual_seed(42)

# Download tiny Shakespeare dataset if not present

data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "/Users/danghuyhoang/Desktop/Standford/deepseek_v3/components/input.txt"

if not os.path.exists(data_path):
    print("Downloading tiny Shakespeare dataset...")
    response = requests.get(data_url)
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Download complete.")
else:
    print("Dataset already exists at", data_path)

class Expert(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
num_experts = 3
top_k = 2
n_embed = 8

mh_output = torch.randn(1, 4, n_embed)
topkgate_linear = nn.Linear(n_embed, num_experts)
logits = topkgate_linear(mh_output)
print(logits)

top_k_logits, top_k_indices = logits.topk(top_k, dim=-1) # ger topk experts
print(top_k_logits)
print(top_k_indices)


zeros = torch.full_like(logits, float('-inf'))
sparse_logits = zeros.scatter(-1, top_k_indices, top_k_logits)
print(sparse_logits)

gating_output = F.softmax(sparse_logits, dim=-1)
print(gating_output)


class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.topk = top_k
        self.linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.linear(mh_output)
        top_k_logits, indices = logits.topk(self.topk, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        gating_output = F.softmax(sparse_logits, dim=-1)
        return gating_output, indices


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = TopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output