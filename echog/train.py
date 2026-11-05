"""Portable staged training for the public aligned-ECG protocol (not the paper run recipe)."""
from pathlib import Path
import argparse, collections, hashlib, json
import numpy as np
import torch
import torch.nn.functional as F
from .data import ROOT,EChoGDataset,tensors
from .models import EchoMAR,ReferenceCodec,R8Adapter,build_teacher,CONFIG
from .models.objectives import reconstruction,generator_loss,match_target,masked_bce
from .models.refiner import motion_objective

def run(stage,root=ROOT,out=None,steps=1000,frames=32,seed=42,device='cpu',codec=None,base=None,teacher=None,without_teacher=False):
    if steps<1 or frames<2:raise ValueError('steps >= 1 and frames >= 2 required')
    if stage=='generator' and not codec:raise ValueError('Generator training requires --codec from codec training')
    if stage=='refiner' and not base:raise ValueError('Refiner training requires --base from generator training')
    if stage=='refiner' and not teacher and not without_teacher:raise ValueError('Supply --teacher, or explicitly use --without-teacher for an ablation')
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    ds=EChoGDataset(root,'train');config=json.loads(CONFIG.read_text())
    directory=Path(out or ROOT/'runs'/stage);directory.mkdir(parents=True,exist_ok=True)
    teacher_model=None;renderer=None
    if stage=='codec':model=ReferenceCodec().to(device)
    elif stage=='generator':
        ck=torch.load(codec,map_location='cpu',weights_only=False)
        if ck.get('stage')!='codec':raise ValueError('Not a codec checkpoint')
        renderer=ReferenceCodec().to(device);renderer.load_state_dict(ck['codec']);renderer.eval().requires_grad_(False)
        model=R8Adapter().to(device)
    elif stage=='teacher':model=build_teacher().to(device)
    elif stage=='refiner':
        model=EchoMAR(config).to(device)
        ck=torch.load(base,map_location='cpu',weights_only=False)
        if ck.get('stage')!='generator' and ck.get('format')!='ubicomp_r8_adapt_v3':raise ValueError('Not a base-generator checkpoint')
        model.base.load_state_dict(ck['model']);model.codec.load_state_dict(ck['codec'])
        model.base.requires_grad_(False);model.codec.requires_grad_(False)
        if teacher:
            teacher_model=build_teacher().to(device)
            tk=torch.load(teacher,map_location='cpu',weights_only=False)
            teacher_model.load_state_dict(tk['model']);teacher_model.eval().requires_grad_(False)
        else:config['teacher']=0
    else:raise ValueError('Unknown stage')
    groups=collections.defaultdict(list)
    for i,r in enumerate(ds.records):
        if stage=='teacher' and not any(int(ds.labels[r['record_id']][c+'_mask']) for c in ds.classes):continue
        groups[r['patient_group_id']].append(i)
    patients=sorted(groups);rng=np.random.default_rng(seed)
    if not patients:raise ValueError('No eligible training patients')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['wd'])
    def clip(v):return v[:,:,:min(11,v.shape[2])]
    trace=[]
    for step in range(steps):
        pid=patients[int(rng.integers(len(patients)))];indices=groups[pid];idx=indices[int(rng.integers(len(indices)))]
        inp,sup,meta=tensors(ds.sample(idx,draw=step,frames=frames),device)
        model.train();opt.zero_grad(set_to_none=True)
        if stage=='codec':
            result,latent,reg=model(inp['reference'],sup['video'])
            loss,_=reconstruction(result['video'],sup['video'],inp['frame_t']);loss=loss+model.base.config.kl_weight*reg
        elif stage=='generator':
            with torch.no_grad():latent,_=renderer.encode(sup['video'],inp['reference'])
            result=model(**inp);pred=renderer.decode(inp['reference'],result['latent'])['video']
            loss,_=generator_loss(result,pred,latent,inp,sup,'no_matching')
        elif stage=='teacher':
            features=model.video_encoder(model.video_adapter(clip(sup['video'])))
            logits=model.classifier(features);loss=masked_bce(logits,sup['labels'],sup['label_mask'])
        else:
            result=model(**inp);pred=result['video'];target=sup['video']
            loss,_=reconstruction(pred,target,inp['frame_t'])
            support=(target.amax(2,keepdim=True)>.03).float()
            fg=((pred-target).abs()*support).sum()/(support.sum()*pred.shape[2]).clamp_min(1)
            edge=(pred.diff(dim=3)-target.diff(dim=3)).abs().mean()+(pred.diff(dim=4)-target.diff(dim=4)).abs().mean()
            coarse=F.avg_pool3d(pred,(1,16,16))-F.avg_pool3d(target,(1,16,16))
            loss=loss+config['fg']*fg+config['edges']*edge+config['coarse']*coarse.abs().mean()+config['motion']*motion_objective(pred,target,inp['frame_t'])
            loss=loss+config['anchor']*F.kl_div(result['match_logits'].log_softmax(-1),match_target(target,inp['reference']),reduction='batchmean')
            loss=loss+config['flow_tv']*(result['flow'].diff(dim=3).abs().mean()+result['flow'].diff(dim=4).abs().mean())
            if teacher_model is not None:
                with torch.no_grad():rt=teacher_model.video_encoder(teacher_model.video_adapter(clip(target)));tl=teacher_model.classifier(rt)
                pf=teacher_model.video_encoder(teacher_model.video_adapter(clip(pred)));pl=teacher_model.classifier(pf);mask=sup['label_mask']
                kd=(F.smooth_l1_loss(pl,tl,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
                loss=loss+config['teacher']*(config['logit']*kd+config['feature']*(1-F.cosine_similarity(pf,rt)).mean())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        row=dict(step=step+1,stage=stage,loss=float(loss.detach()),gradient_norm=float(norm),record_id=meta['record_id'],split='train')
        trace.append(row)
        if step==0 or (step+1)%25==0 or step+1==steps:print(json.dumps(row),flush=True)
    ck=dict(stage=stage,format='echog-aligned-v1',steps=steps,seed=seed,config=config,trained_on='train',frames=frames,
        protocol_sha256=hashlib.sha256((Path(root)/'metadata/protocol.json').read_bytes()).hexdigest(),
        initialization='random' if stage in ('codec','teacher') else 'supplied previous-stage checkpoint',teacher_used=teacher_model is not None)
    if stage=='codec':ck['codec']=model.state_dict()
    else:ck['model']=model.state_dict()
    if renderer is not None:ck['codec']=renderer.state_dict()
    path=directory/'latest.pt';torch.save(ck,path)
    (directory/'training_log.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['codec','generator','teacher','refiner'],required=True)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--frames',type=int,default=32);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu');p.add_argument('--codec');p.add_argument('--base');p.add_argument('--teacher');p.add_argument('--without-teacher',action='store_true')
    print(run(**vars(p.parse_args())))

if __name__=='__main__':main()
