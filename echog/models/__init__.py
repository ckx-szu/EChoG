from pathlib import Path
import json
import torch
from torch import nn
from .codec import ReferenceCodec
from .generator import R8Adapter
from .refiner import Refiner

CONFIG=Path(__file__).resolve().parents[2]/'configs/echomar.json'

class EchoMAR(nn.Module):
    """Portable original model architecture; weights must be trained or supplied."""
    def __init__(self,config=None):
        super().__init__()
        self.config=config or json.loads(CONFIG.read_text())
        self.base=R8Adapter();self.codec=ReferenceCodec()
        self.refiner=Refiner('bandlimited_motion_teacher',self.config)
        self.freeze_base=True
    def train(self,mode=True):
        super().train(mode)
        if self.freeze_base:self.base.eval();self.codec.eval()
        return self
    def forward(self,reference,ecg,ecg_t,frame_t):
        inp=dict(reference=reference,ecg=ecg,ecg_t=ecg_t,frame_t=frame_t)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_base):
            base=self.codec.decode(reference,self.base(**inp)['latent'])['video']
        return self.refiner(**inp,base=base)

def load_model(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('stage')!='refiner' and ck.get('format')!='echomar_search_20261102':
        raise ValueError('Expected a trained refiner or original final EchoMAR checkpoint')
    model=EchoMAR(ck.get('config'));model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval(),ck

def build_teacher():
    from .disease_components import FastDICOMBaseline
    return FastDICOMBaseline('echo',False,.2,None,'kinetics',8,6,'mean',5,'mean',True)
