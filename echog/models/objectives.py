import torch
import torch.nn.functional as F

def ssim_frames(a,b):
    a=a.transpose(1,2).flatten(0,1).float();b=b.transpose(1,2).flatten(0,1).float()
    ma=F.avg_pool2d(a,7,1,3);mb=F.avg_pool2d(b,7,1,3)
    va=F.avg_pool2d(a*a,7,1,3)-ma*ma;vb=F.avg_pool2d(b*b,7,1,3)-mb*mb
    cov=F.avg_pool2d(a*b,7,1,3)-ma*mb
    return (((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean((1,2,3))

def reconstruction(pred,target,times):
    pixel=(pred-target).abs().mean();structural=1-ssim_frames(pred,target).mean()
    dt=times.diff(dim=1).clamp_min(1e-5)[:,None,:,None,None]
    # Velocity error expressed as a 20-ms image change, not warped cycle time.
    velocity=((pred.diff(dim=2)-target.diff(dim=2))*.02/dt).abs().mean()
    mean_p=pred.mean((3,4),keepdim=True);mean_t=target.mean((3,4),keepdim=True)
    amplitude=(pred.std(dim=2,unbiased=False)-target.std(dim=2,unbiased=False)).abs().mean()
    total=pixel+.25*structural+.5*velocity+.1*amplitude
    return total,dict(pixel=pixel,ssim=1-structural,velocity=velocity,amplitude=amplitude)

def match_target(target,reference):
    # Similar-looking beats are valid alternatives; no single-frame one-hot claim.
    a=F.avg_pool2d(target.transpose(1,2).flatten(0,1),8)
    r=F.avg_pool2d(reference,8)
    distance=(a-r).abs().mean((1,2,3))[None]
    return (-distance/.02).softmax(-1).detach()

def generator_loss(out,pred,latent_target,inp,sup,variant):
    total,metrics=reconstruction(pred,sup['video'],inp['frame_t'])
    latent=F.smooth_l1_loss(out['latent'][:,:48],latent_target[:,:48])
    flow=F.smooth_l1_loss(out['latent'][:,48:]/.3,latent_target[:,48:]/.3)
    target=match_target(sup['video'],inp['reference'])
    matching=F.kl_div(out['match_logits'].log_softmax(-1),target,reduction='batchmean')
    # Ordered region motion relative to reference: spatial information retained.
    def signature(x):return F.avg_pool3d(x,(1,16,16),(1,16,16))
    pp=signature(pred-inp['reference'][:,:,None]);tt=signature(sup['video']-inp['reference'][:,:,None])
    ordered=F.smooth_l1_loss(pp/.1,tt/.1)+F.smooth_l1_loss(pp.diff(dim=2)/.05,tt.diff(dim=2)/.05)
    total=total+.1*latent+.1*flow+.05*ordered+(.02 if variant=='matching' else 0)*matching
    return total,{**metrics,'latent':latent,'coarse_flow':flow,'match_kl':matching,'ordered':ordered}

def masked_bce(logits,labels,mask,pos_weight=None):
    values=F.binary_cross_entropy_with_logits(logits,labels,reduction='none',pos_weight=pos_weight)
    return (values*mask).sum()/mask.sum().clamp_min(1)

@torch.no_grad()
def reconstruction_metrics(pred,target,reference,times,ref_index):
    valid=torch.ones(target.shape[2],dtype=torch.bool,device=target.device);valid[ref_index]=False
    mse=(pred-target).square().mean((0,1,3,4));mae=(pred-target).abs().mean((0,1,3,4))
    static=(reference[:,:,None]-target).abs().mean((0,1,3,4))
    # Evaluation-only target support; never fed to the generator.
    foreground=(target.amax(dim=2,keepdim=True)>.03).float()
    fg=((pred-target).abs()*foreground).sum((0,1,3,4))/foreground.sum((0,1,3,4)).clamp_min(1)
    dt=times.diff(dim=1).clamp_min(1e-5)[:,None,:,None,None]
    vel=((pred.diff(dim=2)-target.diff(dim=2))*.02/dt).abs().mean()
    return dict(mae=float(mae[valid].mean()),mse=float(mse[valid].mean()),psnr=float((-10*torch.log10(mse[valid].clamp_min(1e-10))).mean()),
        ssim=float(ssim_frames(pred,target)[valid].mean()),static_mae=float(static[valid].mean()),velocity_error=float(vel),
        reference_mae=float(mae[ref_index]),foreground_mae=float(fg[valid].mean()),motion_mae=float((pred.std(dim=2,unbiased=False)-target.std(dim=2,unbiased=False)).abs().mean()))
