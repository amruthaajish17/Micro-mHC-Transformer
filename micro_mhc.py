import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_x + self.eps) * self.scale

class StandardBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.Linear(d_model, d_model) 
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class UnconstrainedHCBlock(nn.Module):
    def __init__(self, d_model, n=4):
        super().__init__()
        self.n = n
        self.d_model = d_model
        self.stream_norm = RMSNorm(n * d_model)
        self.proj_pre = nn.Linear(n * d_model, n)
        self.proj_post = nn.Linear(n * d_model, n)
        self.proj_res = nn.Linear(n * d_model, n * n)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        self.b_pre = nn.Parameter(torch.ones(n) / n)
        self.b_post = nn.Parameter(torch.ones(n))
        self.b_res = nn.Parameter(torch.eye(n).flatten())
        self.layer_norm = RMSNorm(d_model)
        self.attn = nn.Linear(d_model, d_model) 

    def forward(self, x):
        B, S, n, C = x.shape
        x_flat = x.view(B, S, n * C)
        x_norm = self.stream_norm(x_flat)
        
        H_pre = self.alpha_pre * torch.tanh(self.proj_pre(x_norm)) + self.b_pre
        H_post = self.alpha_post * torch.tanh(self.proj_post(x_norm)) + self.b_post
        H_res = self.alpha_res * torch.tanh(self.proj_res(x_norm)) + self.b_res
        H_res = H_res.view(B, S, n, n)
        
        layer_in = torch.einsum('bsn,bsnc->bsc', H_pre, x)
        layer_out = self.attn(self.layer_norm(layer_in))
        scattered_out = torch.einsum('bsn,bsc->bsnc', H_post, layer_out)
        mixed_res = torch.einsum('bsnm,bsmc->bsnc', H_res, x)
        
        return mixed_res + scattered_out, H_res 

# --- NEW: Day 4 Code Begins Here ---

def sinkhorn_knopp(M, num_iters=20):
    """Projects a matrix onto the Birkhoff polytope (doubly stochastic matrix)."""
    # 1. Make all elements positive using exponentiation
    M = torch.exp(M)
    
    # 2. Iteratively normalize rows and columns
    for _ in range(num_iters):
        M = M / M.sum(dim=-2, keepdim=True) # Column normalization
        M = M / M.sum(dim=-1, keepdim=True) # Row normalization
    return M

class mHCBlock(nn.Module):
    """Manifold-Constrained Hyper-Connections (mHC) Block."""
    def __init__(self, d_model, n=4):
        super().__init__()
        self.n = n
        self.d_model = d_model
        
        self.stream_norm = RMSNorm(n * d_model)
        self.proj_pre = nn.Linear(n * d_model, n)
        self.proj_post = nn.Linear(n * d_model, n)
        self.proj_res = nn.Linear(n * d_model, n * n)
        
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        
        self.b_pre = nn.Parameter(torch.ones(n) / n)
        self.b_post = nn.Parameter(torch.ones(n))
        self.b_res = nn.Parameter(torch.eye(n).flatten())
        
        self.layer_norm = RMSNorm(d_model)
        self.attn = nn.Linear(d_model, d_model) 

    def forward(self, x):
        B, S, n, C = x.shape
        x_flat = x.view(B, S, n * C)
        x_norm = self.stream_norm(x_flat)
        
        raw_pre = self.alpha_pre * self.proj_pre(x_norm) + self.b_pre
        raw_post = self.alpha_post * self.proj_post(x_norm) + self.b_post
        raw_res = self.alpha_res * self.proj_res(x_norm) + self.b_res
        
        # --- THE mHC CONSTRAINTS ---
        # 1. Constrain pre and post mappings to be non-negative using Sigmoid
        H_pre = torch.sigmoid(raw_pre)
        H_post = 2 * torch.sigmoid(raw_post)
        
        # 2. Constrain the residual mapping using Sinkhorn-Knopp
        raw_res = raw_res.view(B, S, n, n)
        H_res = sinkhorn_knopp(raw_res, num_iters=20)
        # ---------------------------
        
        layer_in = torch.einsum('bsn,bsnc->bsc', H_pre, x)
        layer_out = self.attn(self.layer_norm(layer_in))
        
        scattered_out = torch.einsum('bsn,bsc->bsnc', H_post, layer_out)
        mixed_res = torch.einsum('bsnm,bsmc->bsnc', H_res, x)
        
        x_next = mixed_res + scattered_out
        return x_next, H_res

# --- Testing the mHC Block ---
if __name__ == "__main__":
    C = 256
    n = 4
    seq_len = 32
    batch_size = 4
    
    x_expanded = torch.randn(batch_size, seq_len, n, C)
    
    mhc_model = mHCBlock(d_model=C, n=n)
    out_mhc, H_res_mhc = mhc_model(x_expanded)
    
    print("\n--- Constrained mHC Block ---")
    print(f"Output shape: {out_mhc.shape}")
    
    # Prove the Sinkhorn-Knopp algorithm worked
    # The sum of any row or column in H_res should be exactly 1.0
    sample_matrix = H_res_mhc[0, 0] # First batch, first token
    
    print("\nMath Proof (Target = 1.0):")
    print(f"Row sums: {sample_matrix.sum(dim=1).detach().numpy()}")
    print(f"Column sums: {sample_matrix.sum(dim=0).detach().numpy()}")