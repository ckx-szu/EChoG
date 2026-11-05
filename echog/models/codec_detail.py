import torch
from torch import nn
import torch.nn.functional as F
from .codec_base import warp_first
def resize(x, size):
    return F.interpolate(x, size=size, mode='trilinear', align_corners=False)


class DetailRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent = nn.Conv3d(48, 8, 1)
        self.local = nn.Sequential(nn.Conv3d(16, 32, (1,3,3), padding=(0,1,1)),
                                   nn.GroupNorm(8,32), nn.SiLU())
        self.context = nn.Sequential(nn.Conv3d(32,32,(1,3,3),stride=(1,2,2),padding=(0,1,1)),
                                    nn.GroupNorm(8,32),nn.SiLU(),
                                    nn.Conv3d(32,32,(3,1,1),padding=(1,0,0),groups=32),
                                    nn.Conv3d(32,32,1),nn.SiLU(),
                                    nn.Conv3d(32,32,(1,3,3),padding=(0,1,1)),nn.SiLU())
        self.output = nn.Conv3d(32, 3, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, first, latent, out):
        t = out['video'].shape[2]
        size = (t,64,64)
        frame = first[:,:,None].expand(-1,-1,t,-1,-1)
        dx = F.pad(frame.diff(dim=4), (0,1,0,0,0,0))
        dy = F.pad(frame.diff(dim=3), (0,0,0,1,0,0))
        texture = out['texture_effect']
        coarse = out['flow'].permute(0,2,1,3,4)
        x = torch.cat([resize(v,size) for v in
                       (frame,out['video'],out['warped'],texture,coarse,dx,dy,self.latent(latent[:,:48]))],1)
        local = self.local(x)
        value = self.output(local + resize(self.context(local),size))
        fine = .04 * value[:,:2].tanh()
        # Do not replace the successful low-resolution cardiac displacement.
        fine = fine - resize(F.avg_pool3d(fine,(1,4,4)),size)
        texture_delta = .04 * value[:,2:3].tanh()
        full_size = (t,*first.shape[-2:])
        fine, texture_delta = resize(fine,full_size), resize(texture_delta,full_size)
        flow = (coarse + fine).permute(0,2,1,3,4)
        warped = warp_first(first,flow).permute(0,2,1,3,4)
        video = (warped + texture + texture_delta).clamp(0,1)
        return {**out,'video':video,'flow':flow,'warped':warped,
                'fine_flow':fine,'detail_texture':texture_delta}


