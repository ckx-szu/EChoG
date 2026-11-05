from pathlib import Path
import argparse,collections,csv,hashlib,json
import cv2
import numpy as np
from .data import ROOT,EChoGDataset,rows,tensors

def verify(root,full=False):
    rr=rows(root/'metadata/records.csv');assert len(rr)==3264
    identity=collections.defaultdict(set);pixels=collections.defaultdict(set)
    expected={'train':(2249,947),'val':(481,205),'test':(534,199)}
    for r in rr:
        identity[r['patient_group_id']].add(r['split']);pixels[r['video_sha256']].add(r['split'])
        for stem in ['video','ecg','reference','mask']:
            p=root/r[stem+'_path']
            if not p.is_file():raise FileNotFoundError(f'{p}: extract the data archive into the project root')
            if full:assert hashlib.sha256(p.read_bytes()).hexdigest()==r[stem+'_sha256'],str(p)
    assert all(len(x)==1 for x in identity.values()) and all(len(x)==1 for x in pixels.values())
    decoded=0
    for split,(n,g) in expected.items():
        d=EChoGDataset(root,split);assert len(d)==n and len({r['patient_group_id'] for r in d.records})==g
        assert (root/'splits'/f'{split}.txt').read_text().splitlines()==[r['record_id'] for r in d.records]
        for setting in ['diagnosis_only','history_plus_diagnosis']:
            labs=EChoGDataset(root,split,setting).labels
            for r in d.records:
                lab=labs[r['record_id']];assert lab['split']==split
                for c in d.classes:assert int(lab[c]) in (0,1) and int(lab[c+'_mask']) in (0,1)
        for i in range(len(d)) if full else [0]:
            r,v,x,t=d.read_record(i);ref=int(r['reference_frame_zero_based'])
            assert np.array_equal(v[ref],cv2.imread(str(root/r['reference_path']),cv2.IMREAD_GRAYSCALE))
            assert len(v)==len(x)==len(t) and abs(t[0])<1e-6
            decoded+=1
        if split!='train':
            for i,r in enumerate(d.records):
                w=d.windows[r['record_id']];a,b=int(w['start']),int(w['stop']);ref=int(r['reference_frame_zero_based'])
                assert 0<=a<=ref<b<=int(r['frame_count']) and b-a==min(32,int(r['frame_count']))
        print(f'{split}: {n} records, {g} patient groups',flush=True)
    return dict(status='PASS',records=3264,decoded_records=decoded,all_data_hashes_checked=full,no_patient_overlap=True,no_identical_video_overlap=True)

def contact_sheet(video,reference,path):
    ids=np.linspace(0,len(video)-1,min(6,len(video))).round().astype(int)
    images=[reference]+[video[i] for i in ids]
    canvas=np.concatenate(images,axis=1)
    if not cv2.imwrite(str(path),canvas):raise IOError(path)

def infer(root,split,record_id,checkpoint,out,device):
    import torch
    from .models import load_model
    torch.set_num_threads(2)
    d=EChoGDataset(root,split);index=d.index[record_id] if record_id else 0
    inp,sup,meta=tensors(d.sample(index),device)
    if checkpoint:
        model,ck=load_model(checkpoint,device)
        with torch.no_grad():pred=model(**inp)['video']
        kind='EchoMAR checkpoint'
    else:pred=inp['reference'][:,:,None].expand_as(sup['video']);kind='Repeated reference baseline'
    video=(pred[0,0].clamp(0,1).cpu().numpy()*255).round().astype('uint8')
    out.mkdir(parents=True,exist_ok=True)
    times=inp['frame_t'][0].cpu().numpy();fps=float(1/np.median(np.diff(times)))
    writer=cv2.VideoWriter(str(out/'prediction.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(128,128),isColor=False)
    if not writer.isOpened():raise RuntimeError('MP4 encoder unavailable')
    for frame in video:writer.write(frame)
    writer.release()
    contact_sheet(video,(inp['reference'][0,0].cpu().numpy()*255).astype('uint8'),out/'preview.png')
    # Reference index is used only for scoring, never forwarded to the model.
    valid=np.ones(len(video),dtype=bool);valid[meta['reference_in_window']]=False
    mae=float((pred-sup['video']).abs()[0,0,torch.from_numpy(valid).to(device)].mean())
    result=dict(**meta,method=kind,nonreference_mae=mae,frame_times=times.tolist(),protocol='echog-aligned-v1',paper_comparable=False)
    (out/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return result

def smoke(root,out,device):
    from .train import run
    verify(root)
    d=EChoGDataset(root,'val')
    a=d.sample(0,draw=0);b=d.sample(0,draw=999)
    assert a[2]==b[2] and all(np.array_equal(a[0][k],b[0][k]) for k in a[0])
    assert set(a[0])=={'reference','ecg','ecg_t','frame_t'}
    paths={}
    for stage in ['codec','generator','teacher','refiner']:
        paths[stage]=run(stage,root=root,out=out/stage,steps=1,frames=8,device=device,codec=paths.get('codec'),base=paths.get('generator'),teacher=paths.get('teacher'))
    infer(root,'val',None,paths['refiner'],out/'inference',device)
    report=dict(status='PASS',stages=list(paths),steps_per_stage=1,device=device,trained_weights_for_quality_evaluation=False)
    (out/'smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

def main():
    p=argparse.ArgumentParser(description='EChoG public data and EchoMAR portable runner')
    p.add_argument('command',choices=['verify','demo','infer','smoke'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--full',action='store_true')
    p.add_argument('--split',choices=['train','val','test'],default='val');p.add_argument('--record-id')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--out',type=Path,default=ROOT/'outputs/demo')
    a=p.parse_args()
    if a.command=='verify':
        result=verify(a.root,a.full);print(json.dumps(result,indent=2))
        a.out.mkdir(parents=True,exist_ok=True);(a.out/'verification.json').write_text(json.dumps(result,indent=2))
    elif a.command=='smoke':smoke(a.root,a.out,a.device)
    elif a.command=='infer' and not a.checkpoint:p.error('infer requires --checkpoint; use demo for the repeated-reference baseline')
    else:infer(a.root,a.split,a.record_id,a.checkpoint if a.command=='infer' else None,a.out,a.device)

if __name__=='__main__':main()
