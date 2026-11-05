import torch
from torch import nn
import torch.nn.functional as F
from .codec_base import warp_first
def resize(x,size):
    return F.interpolate(x,size=size,mode='trilinear',align_corners=False)

def advect(sequence,flow):
    """Warp time-varying canonical appearance using normalized backward maps."""
    b,c,t,h,w=sequence.shape
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=sequence.device),
                        torch.linspace(-1,1,w,device=sequence.device),indexing='ij')
    grid=torch.stack((xx,yy),-1)[None,None]+flow.float().permute(0,1,3,4,2)
    values=sequence.float().permute(0,2,1,3,4).reshape(b*t,c,h,w)
    result=F.grid_sample(values,grid.reshape(b*t,h,w,2),mode='bilinear',padding_mode='border',align_corners=True)
    return result.reshape(b,t,c,h,w).permute(0,2,1,3,4)

class MaterialTransport(nn.Module):
    def __init__(self):
        super().__init__()
        self.anatomy=nn.Sequential(nn.Conv2d(1,16,3,padding=1,stride=2),nn.GroupNorm(4,16),nn.SiLU(),
                                  nn.Conv2d(16,8,3,padding=1),nn.SiLU())
        self.latent=nn.Conv3d(48,16,1)
        self.features=nn.Sequential(nn.Conv3d(26,32,(1,3,3),padding=(0,1,1)),nn.GroupNorm(8,32),nn.SiLU(),
                                   nn.Conv3d(32,32,(3,1,1),padding=(1,0,0),groups=32),nn.Conv3d(32,32,1),nn.SiLU(),
                                   nn.Conv3d(32,32,(1,3,3),padding=(0,2,2),dilation=(1,2,2)),nn.SiLU())
        self.output=nn.Conv3d(32,3,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,first,latent,out):
        t=out['video'].shape[2];size=(t,64,64)
        anatomy=self.anatomy(first)[:,:,None].expand(-1,-1,t,-1,-1)
        # Both corrections are represented on the reference-frame coordinate grid.
        coarse=out['flow'].permute(0,2,1,3,4)
        x=torch.cat((anatomy,resize(self.latent(latent[:,:48]),size),resize(coarse,size)),1)
        raw=self.output(self.features(x))
        delta=.03*raw[:,:2].tanh()
        material=.04*raw[:,2:].tanh()
        full=(t,*first.shape[-2:])
        delta,material=resize(delta,full),resize(material,full)
        flow=(coarse+delta).permute(0,2,1,3,4)
        warped=warp_first(first,flow).permute(0,2,1,3,4)
        carried=advect(material,flow)
        # Preserve the inherited trained appearance; add geometry and transported
        # detail separately. Exact zero-start in residual arithmetic.
        video=(out['video']+(warped-out['warped'])+carried).clamp(0,1)
        return {**out,'video':video,'flow':flow,'warped':warped,
                'material_flow':delta,'material_texture':carried}

