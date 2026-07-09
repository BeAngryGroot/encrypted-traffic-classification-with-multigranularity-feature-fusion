# models/mamba_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.eps=eps; self.w=nn.Parameter(torch.ones(d))
    def forward(self, x):
        rms=(x.pow(2).mean(-1,keepdim=True)+self.eps).sqrt(); return x/rms*self.w

# 优先用官方 mamba-ssm；无则回退到轻量实现
USE_OFFICIAL=False
try:
    from mamba_ssm import Mamba2
    USE_OFFICIAL=True
except Exception:
    USE_OFFICIAL=False

class SimpleSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=64, conv_kernel=3, dropout=0.1):
        super().__init__()
        self.dw=nn.Conv1d(d_model,d_model,conv_kernel,groups=d_model,padding=(conv_kernel - 1 )//2)
        self.inp=nn.Linear(d_model,2*d_model)
        self.A=nn.Parameter(torch.randn(d_state,d_state)*0.01)
        self.B=nn.Parameter(torch.randn(d_state,d_model)*0.01)
        self.C=nn.Parameter(torch.randn(d_model,d_state)*0.01)
        self.out=nn.Linear(d_model,d_model)
        self.drop=nn.Dropout(dropout); self.norm=RMSNorm(d_model)
    def forward(self,x,mask=None): # x:(B,L,D)
        r=x; 
        x=self.norm(x); 
        x=self.dw(x.transpose(1,2)).transpose(1,2)
        u,v=self.inp(x).chunk(2,-1); 
        v=torch.sigmoid(v)
        B,L,D=u.shape; 
        h=torch.zeros(B,self.A.size(0),device=x.device,dtype=x.dtype)
        ys=[]

        if mask is not None and mask.size(1) == 1 and L > 1:
            mask = mask.expand(-1, L)            # (B, L)

        for t in range(L):
            h = h@self.A.T + u[:,t,:]@self.B.T
            y = h@self.C.T
            if mask is not None: 
                y = y*mask[:,t].unsqueeze(-1)
                ys.append(y)
        y=torch.stack(ys,1); 
        y=y*v; 
        y=self.out(y)
        return r+self.drop(y)

class AttnPool(nn.Module):
    def __init__(self, d, temperature: float = 1.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d))
        self.temperature = temperature

    def forward(self, x, mask=None):
        s = x @ self.q
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e4)  # ✅ 修复溢出
        s = torch.clamp(s, min=-1e4, max=1e4)       # ✅ 防溢出
        s = s - s.max(dim=1, keepdim=True).values
        w = torch.softmax(s / self.temperature, dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)

class MambaStack(nn.Module):
    def __init__(self,d_model,n_layers=5,d_state=64,dropout=0.2):
        super().__init__();
        self.blks=nn.ModuleList([(Mamba2(d_model=d_model,d_state=d_state,expand=2) if USE_OFFICIAL else SimpleSSMBlock(d_model,d_state,dropout=dropout)) for _ in range(n_layers)])
        self.norm=RMSNorm(d_model)
    def forward(self,x,mask=None):
        for b in self.blks:
            x = b(x) if not isinstance(b,SimpleSSMBlock) else b(x,mask)
        return self.norm(x)

class MicroMambaBranch(nn.Module):
    def __init__(self, d_in, d_model=256, n_layers=3, d_state=64, dropout=0.1):
        super().__init__()
        self.emb = nn.Linear(d_in, d_model)
        self.backbone = MambaStack(d_model, n_layers, d_state, dropout)
        self.pool = AttnPool(d_model)  # ← 用上面的稳定版

    def forward(self, x, mask=None):  # x:(B,T,D_micro), mask:(B,T) 1/0
        x = self.emb(x)
        x = self.backbone(x, mask)    # 如果 Mamba2 不吃 mask，也没关系，pool 会吃
        return self.pool(x, mask)
