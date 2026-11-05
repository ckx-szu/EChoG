"""Evaluate fixed public windows; image scores exclude the supplied reference frame."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from .data import ROOT,EChoGDataset,tensors
from .models import load_model
from .models.objectives import reconstruction_metrics

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/evaluation.json');a=p.parse_args()
    torch.set_num_threads(2)
    d=EChoGDataset(a.root,a.split);model=load_model(a.checkpoint,a.device)[0] if a.checkpoint else None
    output=[]
    for i in range(min(a.limit,len(d)) if a.limit>0 else len(d)):
        inp,sup,meta=tensors(d.sample(i),a.device)
        pred=model(**inp)['video'] if model is not None else inp['reference'][:,:,None].expand_as(sup['video'])
        metrics=reconstruction_metrics(pred,sup['video'],inp['reference'],inp['frame_t'],meta['reference_in_window'])
        output.append(dict(**meta,**metrics))
        if (i+1)%100==0:print(f'Evaluated {i+1}/{len(d)}',flush=True)
    result=dict(protocol='echog-aligned-v1',paper_comparable=False,method='EchoMAR' if model is not None else 'repeat_reference',records=len(output),partial=len(output)!=len(d),
        summary={k:float(np.mean([r[k] for r in output])) for k in metrics},rows=output,
        scope='Image and motion metrics only; disease AP and ECG intervention analyses are not computed by this command.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
