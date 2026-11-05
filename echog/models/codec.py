import hashlib,math
import torch
from torch import nn
import torch.nn.functional as F
from .codec_base import GenerationFriendlyMotionCodec,MotionCodecV2Config,warp_first
from .codec_detail import DetailRenderer
from .codec_transport import MaterialTransport

class ReferenceCodec(nn.Module):
    def __init__(self):
        super().__init__();self.base=GenerationFriendlyMotionCodec(MotionCodecV2Config(temporal_stride=1))
        self.detail=DetailRenderer();self.transport=MaterialTransport();self.exposure='new_split_random_initialization'
    def load_r8(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False);s=state['codec']
        if state['variant']!='transport_balanced':raise ValueError('Expected audited R8 transport_balanced checkpoint')
        loaded={}
        for module,prefix in [(self.base,'parent.parent.base.'),(self.detail,'parent.detail.'),(self.transport,'transport.')]:
            subset={k[len(prefix):]:v for k,v in s.items() if k.startswith(prefix)}
            module.load_state_dict(subset,strict=True);loaded[prefix]=len(subset)
        self.exposure='R8_legacy_pretraining_overlap_EXPLORATORY'
        return dict(path=str(path),sha256=hashlib.sha256(open(path,'rb').read()).hexdigest(),variant=state['variant'],step=state['step'],loaded=loaded,exposure=self.exposure,
            omitted='legacy frozen reference encoder copy and legacy ECG predictor; no new initialization weights silently ignored')
    def encode(self,video,reference):
        posterior=self.base.encode(video,reference)
        # Relative to this image's self-encoding, never relative to output frame 0.
        zero=self.base.encode(reference[:,:,None],reference)['latent']
        latent=posterior['latent']-zero
        decoded=self.base.decode(reference,latent,video.shape[2])
        flow=F.interpolate(decoded['flow'].permute(0,2,1,3,4),size=latent.shape[2:],mode='trilinear',align_corners=False)
        return torch.cat((latent,flow),1),posterior['regularization']
    def decode(self,reference,latent):
        t=latent.shape[2];out=self.base.decode(reference,latent[:,:48],t)
        flow=F.interpolate(latent[:,48:],size=(t,128,128),mode='trilinear',align_corners=False).permute(0,2,1,3,4)
        warped=warp_first(reference,flow).permute(0,2,1,3,4)
        texture=out['video']-out['warped']
        out={**out,'flow':flow,'warped':warped,'texture_effect':texture,'video':(warped+texture).clamp(0,1)}
        out=self.detail(reference,latent,out);return self.transport(reference,latent,out)
    def forward(self,reference,video):
        latent,reg=self.encode(video,reference);return self.decode(reference,latent),latent,reg


def time_features(t):
    freq=2.**torch.arange(-3,5,device=t.device,dtype=t.dtype)
    phase=2*math.pi*t[...,None]*freq
    return torch.cat((t[...,None],phase.sin(),phase.cos()),-1)

