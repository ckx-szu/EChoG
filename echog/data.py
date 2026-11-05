from pathlib import Path
import csv, hashlib, json
import cv2
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

class EChoGDataset:
    """Full-record I/O plus deterministic validation/test samples. No silent filtering."""
    def __init__(self,root=ROOT,split='train',label_setting='diagnosis_only'):
        self.root=Path(root).resolve()
        if split not in ('train','val','test'):raise ValueError('Unknown split')
        if label_setting not in ('diagnosis_only','history_plus_diagnosis'):raise ValueError('Unknown labels')
        self.split=split
        self.records=[r for r in rows(self.root/'metadata/records.csv') if r['split']==split]
        self.index={r['record_id']:i for i,r in enumerate(self.records)}
        self.labels={r['record_id']:r for r in rows(self.root/f'metadata/labels_{label_setting}.csv')}
        self.classes=[r['id'] for r in json.loads((self.root/'metadata/classes.json').read_text(encoding='utf-8'))]
        self.windows={r['record_id']:r for r in rows(self.root/'metadata/evaluation_windows.csv')}
    def __len__(self):return len(self.records)
    def read_record(self,index):
        r=self.records[index] if isinstance(index,int) else self.records[self.index[index]]
        cap=cv2.VideoCapture(str(self.root/r['video_path']));frames=[]
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        finally:cap.release()
        v=np.asarray(frames,dtype=np.uint8)
        m=loadmat(self.root/r['ecg_path']);x=m['ecg'].ravel();t=m['frame_time_seconds'].ravel()
        n=int(r['frame_count'])
        if v.shape!=(n,128,128) or x.shape!=(n,) or t.shape!=(n,):raise ValueError(f'Alignment mismatch: {r["record_id"]}')
        if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Invalid ECG/time')
        return r,v,x,t
    def sample(self,index,draw=0,frames=None,full=False):
        r,v,x,t=self.read_record(index);ref=int(r['reference_frame_zero_based']);n=len(v)
        if not 0<=ref<n:raise ValueError('Invalid reference')
        if full:start,stop=0,n
        elif self.split!='train':
            if frames not in (None,32):raise ValueError('Evaluation uses the supplied fixed windows; use full=True for a separate full-record protocol')
            w=self.windows[r['record_id']];start,stop=int(w['start']),int(w['stop'])
        else:
            length=min(int(frames or 32),n)
            if length<2:raise ValueError('Need at least two frames')
            lo=max(0,ref-length+1);hi=min(ref,n-length)
            seed=int.from_bytes(hashlib.sha256(f'20261030|{r["record_id"]}|{draw}'.encode()).digest()[:8],'little')
            start=lo+seed%(hi-lo+1);stop=start+length
        signal=x[start:stop].astype('float32');signal=(signal-signal.mean())/max(float(signal.std()),1e-6)
        clock=(t[start:stop]-t[start]).astype('float32')
        lab=self.labels[r['record_id']]
        inputs=dict(reference=v[ref][None,None].astype('float32')/255,ecg=signal[None],ecg_t=clock[None],frame_t=clock[None])
        targets=dict(video=v[start:stop][None,None].astype('float32')/255,labels=np.array([[int(lab[c]) for c in self.classes]],dtype='float32'),label_mask=np.array([[int(lab[c+'_mask']) for c in self.classes]],dtype='float32'))
        meta=dict(record_id=r['record_id'],patient_group_id=r['patient_group_id'],split=self.split,start=start,stop=stop,reference_index=ref,reference_in_window=ref-start)
        return inputs,targets,meta
    def __getitem__(self,index):return self.sample(index)

def tensors(sample,device='cpu'):
    import torch
    inp,sup,meta=sample
    return {k:torch.from_numpy(v).to(device) for k,v in inp.items()},{k:torch.from_numpy(v).to(device) for k,v in sup.items()},meta
