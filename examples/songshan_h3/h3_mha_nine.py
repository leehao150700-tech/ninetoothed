"""gfx936 native MHA: Nine-generated numerical code; host views only."""
from functools import lru_cache,partial
import math,os
from pathlib import Path
import torch
import ninetoothed as nt
import ninetoothed.language as nl
from ninetoothed.language import libdevice
import ninetoothed.generation
ninetoothed.generation.CACHE_DIR=Path(os.environ["NINETOOTHED_CACHE_DIR"])

def _arrange(q,k,v,scale,out,causal,causal_offset,bm=16,bn=32):
    def query_view(tensor):
        arranged=tensor.tile((1,1,bm,-1)).tile((1,q.shape[-3]//k.shape[-3],1,1))
        arranged.dtype=arranged.dtype.squeeze((0,2,3))
        arranged.dtype.dtype=arranged.dtype.dtype.squeeze((0,1))
        return arranged
    query=query_view(q)
    def key_view(tensor):
        arranged=tensor.tile((1,1,bn,-1)).tile((1,1,-1,-1)).expand((-1,-1,query.shape[-2],-1))
        arranged.dtype=arranged.dtype.squeeze((0,1,3))
        arranged.dtype.dtype=arranged.dtype.dtype.squeeze((0,1))
        return arranged
    return query,key_view(k),key_view(v),scale,query_view(out),causal,causal_offset

def _application(q,k,v,scale,out,causal,causal_offset):
    for i in range(q.shape[0]):
        qi=nl.cast(q[i],nl.float32)
        acc=nl.zeros((qi.shape[-2],qi.shape[-1]),dtype=nl.float32)
        denom=nl.zeros((qi.shape[-2],),dtype=nl.float32)
        maximum=nl.full((qi.shape[-2],),-1.0e30,dtype=nl.float32)
        for j in range((k.source.shape[-2]+k.dtype.shape[0]-1)//k.dtype.shape[0]):
            scores=nl.dot(qi,nl.trans(nl.cast(k[j],nl.float32)),input_precision='ieee')*scale
            key_pos=k[j].offsets(-2)
            scores=nl.where(key_pos<k.source.shape[-2],scores,-1.0e30)
            if causal:
                query_pos=q[i].offsets(-2)
                scores=nl.where(query_pos[:,None]+causal_offset>=key_pos[None,:],scores,-1.0e30)
            next_max=nl.maximum(maximum,nl.max(scores,1))
            probability=libdevice.exp(nl.maximum(scores-next_max[:,None],-80.0))
            probability=nl.where(key_pos<k.source.shape[-2],probability,0.0)
            if causal:
                probability=nl.where(query_pos[:,None]+causal_offset>=key_pos[None,:],probability,0.0)
            alpha=libdevice.exp(nl.maximum(maximum-next_max,-80.0))
            acc=acc*alpha[:,None]+nl.dot(probability,nl.cast(v[j],nl.float32),input_precision='ieee')
            denom=denom*alpha+nl.sum(probability,1)
            maximum=next_max
        out[i]=acc/nl.maximum(denom[:,None],1.0e-20)

def _application_mixed(q,k,v,scale,out,causal,causal_offset):
    for i in range(q.shape[0]):
        qi=q[i]
        acc=nl.zeros((qi.shape[-2],qi.shape[-1]),dtype=nl.float32)
        denom=nl.zeros((qi.shape[-2],),dtype=nl.float32)
        maximum=nl.full((qi.shape[-2],),-1.0e30,dtype=nl.float32)
        for j in range((k.source.shape[-2]+k.dtype.shape[0]-1)//k.dtype.shape[0]):
            scores=nl.dot(qi,nl.trans(k[j]),input_precision='ieee')*scale
            key_pos=k[j].offsets(-2)
            scores=nl.where(key_pos<k.source.shape[-2],scores,-1.0e30)
            if causal:
                query_pos=q[i].offsets(-2)
                scores=nl.where(query_pos[:,None]+causal_offset>=key_pos[None,:],scores,-1.0e30)
            next_max=nl.maximum(maximum,nl.max(scores,1))
            probability=libdevice.exp(nl.maximum(scores-next_max[:,None],-80.0))
            probability=nl.where(key_pos<k.source.shape[-2],probability,0.0)
            if causal:
                probability=nl.where(query_pos[:,None]+causal_offset>=key_pos[None,:],probability,0.0)
            alpha=libdevice.exp(nl.maximum(maximum-next_max,-80.0))
            acc=acc*alpha[:,None]+nl.dot(probability.to(v[j].dtype),v[j],input_precision='ieee')
            denom=denom*alpha+nl.sum(probability,1)
            maximum=next_max
        out[i]=acc/nl.maximum(denom[:,None],1.0e-20)

@lru_cache(maxsize=64)
def _kernel(dim,causal,bm,bn,mixed=False):
    tensors=[nt.Tensor(4,other=0,shape_options=(None,None,None,{'constexpr':True,'upper_bound':256})) for _ in range(3)]
    scale=nt.Tensor(0,dtype=nt.float32)
    out=nt.Tensor(4,shape_options=(None,None,None,{'constexpr':True,'upper_bound':256}))
    is_causal=nt.Tensor(0,constexpr=True,value=causal)
    causal_offset=nt.Tensor(0,dtype=nt.int32)
    return nt.make(partial(_arrange,bm=bm,bn=bn),_application_mixed if mixed else _application,(*tensors,scale,out,is_causal,causal_offset),kernel_name=f'songshan_h3_attn_d{dim}_c{int(causal)}_m{bm}_n{bn}_offset_mixed{int(mixed)}',num_warps=4,num_stages=1,max_num_configs=1)

def attention(q,k,v,*,out,scale,is_causal=False):
    if any(t.ndim!=4 for t in (q,k,v,out)):
        raise ValueError('Nine MHA expects [B,S,H,D]')
    if q.shape[0]!=k.shape[0] or k.shape!=v.shape or q.shape[-1]!=k.shape[-1]:
        raise ValueError('Incompatible MHA shapes')
    if min(*q.shape,*k.shape)<=0 or q.shape[2]%k.shape[2] or q.shape[-1]>256:
        raise ValueError('Unsupported MHA dimension')
    if any(t.device!=q.device or t.dtype!=q.dtype for t in (k,v,out)) or q.device.type!='cuda':
        raise ValueError('MHA tensors must share a CUDA-compatible device and dtype')
    if q.dtype not in (torch.bfloat16,torch.float16,torch.float32) or out.shape!=q.shape:
        raise ValueError('Unsupported MHA dtype/output')
    if not math.isfinite(scale):raise ValueError('MHA scale must be finite')
    # Establish gfx936 correctness first; do not inherit gfx906 mixed-precision policy.
    kernel=_kernel(q.shape[-1],bool(is_causal),16,32,False)
    kernel(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),
           float(scale),out.permute(0,2,1,3),bool(is_causal),0)
    return out
