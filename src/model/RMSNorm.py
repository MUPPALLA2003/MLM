import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self,config,eps:float= 1e-05,device: torch.device | None = None):

        super().__init__()
        self.num_features = config.num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(config.num_features, device= device, dtype= torch.float32))

    def forward(self,x:torch.Tensor):

        assert x.shape[-1] == self.num_features

        t,dtype = x.float(),x.dtype
        t = t * torch.rsqrt(torch.mean(t**2,dim = -1,keepdim = True) + self.eps)
        t = (t * self.scale).to(dtype)

        return t   