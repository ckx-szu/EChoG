import copy,math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .codec import ReferenceCodec,time_features
from .r8_axial import AxialLatentBlock
from .disease_components import ECGResNet1D

class R8Adapter(nn.Module):
    def __init__(self,variant='no_matching',initialization='clean'):
        super().__init__();self.variant=variant;self.initialization=initialization
        self.reference_encoder=copy.deepcopy(ReferenceCodec().base.first_encoder)
        ecg=ECGResNet1D(.05);self.ecg_stem=ecg.stem;self.ecg_blocks=ecg.blocks
        self.ecg_project=nn.Conv1d(512,768,1);self.context_time=nn.Linear(17,768)
        self.anatomy_projection=nn.Sequential(nn.Conv2d(48,256,1),nn.GroupNorm(16,256),nn.SiLU())
        self.plane_embedding=nn.Parameter(torch.empty(11,256));nn.init.normal_(self.plane_embedding,std=.02)
        self.physical_time=nn.Linear(17,256);nn.init.zeros_(self.physical_time.weight);nn.init.zeros_(self.physical_time.bias)
        self.query_norm=nn.LayerNorm(256);self.context_norm=nn.LayerNorm(768)
        self.ecg_attention=nn.MultiheadAttention(256,8,kdim=768,vdim=768,dropout=.05,batch_first=True)
        self.ecg_scale=nn.Parameter(torch.tensor(.5).atanh())
        self.blocks=nn.ModuleList([AxialLatentBlock(256,8,.05) for _ in range(3)])
        self.output=nn.Conv3d(256,48,1);self.direct=nn.Conv3d(256,2,1)
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.direct.weight,std=.001);nn.init.zeros_(self.direct.bias)
        self.loaded=[]
        if initialization=='legacy':
            raise ValueError('Legacy initialization is unavailable in this release')
    def forward(self,reference,ecg,ecg_t,frame_t):
        assert reference.shape[0]==1
        anatomy=self.reference_encoder(reference);b,_,h,w=anatomy.shape;t=frame_t.shape[1]
        context=self.ecg_project(self.ecg_blocks(self.ecg_stem(ecg[:,None])))
        # Native seconds, no supplied peaks or reference offset. Local 13-token context.
        offsets=torch.linspace(-.6,.6,13,device=ecg.device)
        query_t=frame_t[:,:,None]+offsets
        grid_x=2*(query_t-ecg_t[:,:1,None])/(ecg_t[:,-1:,None]-ecg_t[:,:1,None]).clamp_min(1e-6)-1
        grid=torch.stack((grid_x,torch.zeros_like(grid_x)),-1)
        ctx=F.grid_sample(context[:,:,None],grid,mode='bilinear',padding_mode='border',align_corners=True).permute(0,2,3,1)
        ctx=ctx+self.context_time(time_features(query_t))
        if self.variant=='reference_only':ctx=torch.zeros_like(ctx)
        position=F.interpolate(self.plane_embedding.T[None],size=t,mode='linear',align_corners=True).transpose(1,2)
        position=position+self.physical_time(time_features(frame_t))
        value=self.anatomy_projection(anatomy)[:,:,None]+position.transpose(1,2)[:,:,:,None,None]
        q=value.permute(0,2,3,4,1).reshape(b*t,h*w,256);ctx=self.context_norm(ctx.reshape(b*t,13,768))
        attended,_=self.ecg_attention(self.query_norm(q),ctx,ctx,need_weights=False)
        value=value+self.ecg_scale.tanh()*attended.reshape(b,t,h,w,256).permute(0,4,1,2,3)
        for block in self.blocks:
            value=checkpoint(block,value,use_reentrant=False) if self.training else block(value)
        latent=torch.cat((self.output(value),.3*self.direct(value).tanh()),1)
        dummy=value.mean((1,3,4))
        return dict(latent=latent,match_logits=dummy*0,match_prob=torch.ones_like(dummy)/t)

