import math
import torch
from torch import nn
import torch.nn.functional as F
from .codec import time_features

def block(cin,cout):
    return nn.Sequential(nn.Conv3d(cin,cout,(1,3,3),stride=(1,2,2),padding=(0,1,1)),
        nn.GroupNorm(8,cout),nn.SiLU(),nn.Conv3d(cout,cout,(3,1,1),padding=(1,0,0),groups=cout),
        nn.Conv3d(cout,cout,1),nn.GroupNorm(8,cout),nn.SiLU())


class Refiner(nn.Module):
    def __init__(self,variant,config=None):
        super().__init__();self.variant=variant;self.config=config or {}
        self.ecg=nn.Sequential(nn.Conv1d(1,32,15,2,7),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv1d(32,96,9,2,4),nn.GroupNorm(8,96),nn.SiLU(),
            nn.Conv1d(96,96,5,padding=4,dilation=2),nn.GroupNorm(8,96),nn.SiLU(),
            nn.Conv1d(96,96,5,padding=8,dilation=4),nn.GroupNorm(8,96),nn.SiLU())
        self.time=nn.Linear(17,96);self.reference=nn.Sequential(nn.Conv2d(1,24,5,2,2),nn.SiLU(),nn.Conv2d(24,96,5,2,2),nn.SiLU())
        self.key=nn.Linear(96,96);self.query=nn.Linear(96,96)
        self.film=nn.ModuleList([nn.Linear(192,2*c) for c in [24,48,96]])
        self.enc1=block(3,24);self.enc2=block(24,48);self.enc3=block(48,96)
        self.dec2=nn.Sequential(nn.Conv3d(144,48,(1,3,3),padding=(0,1,1)),nn.GroupNorm(8,48),nn.SiLU(),
            nn.Conv3d(48,48,(3,1,1),padding=(1,0,0),groups=48),nn.SiLU())
        self.dec1=nn.Sequential(nn.Conv3d(72,24,(1,3,3),padding=(0,1,1)),nn.GroupNorm(8,24),nn.SiLU(),
            nn.Conv3d(24,24,(3,1,1),padding=(1,0,0),groups=24),nn.SiLU())
        if self.config.get('shallow'):
            del self.enc3,self.dec2
            self.film=nn.ModuleList(list(self.film)[:2])
        if self.config.get('no_anchor') or self.config.get('no_film'):
            del self.key,self.query,self.reference
        if self.config.get('no_film'):
            del self.ecg,self.time,self.film
        channels=(2 if self.config.get('flow_scale',.08)>0 else 0)+(1 if self.config.get('residual_scale',.12)>0 else 0)
        self.out=nn.Conv3d(24,channels,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,reference,ecg,ecg_t,frame_t,base):
        b,_,n,h,w=base.shape
        logits=torch.zeros(b,n,device=base.device);prob=logits.softmax(-1)
        if not self.config.get('no_film'):
            local=self.ecg(ecg[:,None]);gx=2*(frame_t-ecg_t[:,:1])/(ecg_t[:,-1:]-ecg_t[:,:1]).clamp_min(1e-6)-1
            grid=torch.stack((gx,torch.zeros_like(gx)),-1)[:,None]
            state=F.grid_sample(local[:,:,None],grid,align_corners=True,padding_mode='border').squeeze(2).transpose(1,2)+self.time(time_features(frame_t))
            if self.config.get('no_anchor'):
                cond=torch.cat((state,state),-1)
            else:
                anatomy=self.reference(reference).mean((2,3));logits=(self.key(state)*self.query(anatomy)[:,None]).sum(-1)/math.sqrt(96)
                prob=logits.softmax(-1);anchor=(state*prob[:,:,None]).sum(1)
                cond=torch.cat((state,state-anchor[:,None]),-1)
        def mod(x,i):
            if self.config.get('no_film'):return x
            scale,shift=self.film[i](cond).chunk(2,-1)
            return x*(1+.25*scale.tanh().transpose(1,2)[:,:,:,None,None])+.25*shift.tanh().transpose(1,2)[:,:,:,None,None]
        ref=reference[:,:,None].expand_as(base)
        x1=mod(self.enc1(torch.cat((base,ref,base-ref),1)),0)
        x2=mod(self.enc2(x1),1)
        if self.config.get('shallow'):u2=x2
        else:
            x3=mod(self.enc3(x2),2)
            u2=self.dec2(torch.cat((F.interpolate(x3,size=x2.shape[2:],mode='trilinear',align_corners=False),x2),1))
        u1=self.dec1(torch.cat((F.interpolate(u2,size=x1.shape[2:],mode='trilinear',align_corners=False),x1),1))
        raw=self.out(u1)
        nf=2 if self.config.get('flow_scale',.08)>0 else 0
        flow=self.config.get('flow_scale',.08)*raw[:,:2].tanh() if nf else raw.new_zeros(b,2,n,*raw.shape[-2:])
        residual=self.config.get('residual_scale',.12)*raw[:,nf:nf+1].tanh() if self.config.get('residual_scale',.12)>0 else raw.new_zeros(b,1,n,*raw.shape[-2:])
        sigma=self.config.get('sigma',.04)
        if sigma>0:
            dt=frame_t[:,:,None]-frame_t[:,None,:]
            kernel=torch.exp(-.5*(dt.float()/sigma).square());kernel=kernel/kernel.sum(-1,keepdim=True)
            smooth=torch.einsum('bts,bcshw->bcthw',kernel,torch.cat((flow,residual),1).float())
            flow=smooth[:,:2];residual=smooth[:,2:3]
        size=self.config.get('coarse_size',16)
        residual=F.adaptive_avg_pool3d(residual,(n,size,size))
        flow=F.interpolate(flow,size=base.shape[2:],mode='trilinear',align_corners=False)
        residual=F.interpolate(residual,size=base.shape[2:],mode='trilinear',align_corners=False)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=base.device),torch.linspace(-1,1,w,device=base.device),indexing='ij')
        sampling=torch.stack((xx,yy),-1)[None,None]+flow.permute(0,2,3,4,1).float()
        warped=F.grid_sample(base.transpose(1,2).reshape(b*n,1,h,w).float(),sampling.reshape(b*n,h,w,2),align_corners=True,padding_mode='border')
        video=(warped.reshape(b,n,1,h,w).transpose(1,2)+residual).clamp(0,1)
        return dict(video=video,flow=flow,match_logits=logits,match_prob=prob,anchored=not(self.config.get("no_anchor") or self.config.get("no_film")))


def motion_objective(pred,target,times):
    """Supervised multiscale rates and regional excursion; no smoothing-only reward."""
    total=pred.new_zeros(())
    for scale in [1,4,16]:
        p=pred if scale==1 else F.avg_pool3d(pred,(1,scale,scale))
        t=target if scale==1 else F.avg_pool3d(target,(1,scale,scale))
        for lag in [1,3]:
            if p.shape[2]<=lag:continue
            dt=(times[:,lag:]-times[:,:-lag]).clamp_min(1e-5)[:,None,:,None,None]
            total=total+((p[:,:,lag:]-p[:,:,:-lag]-t[:,:,lag:]+t[:,:,:-lag])*.02/dt).abs().mean()/6
        total=total+.1*(p.std(2,unbiased=False)-t.std(2,unbiased=False)).abs().mean()/3
    return total

