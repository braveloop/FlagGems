# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Load-time registered, lossless down-projection NT weight tiles."""
import importlib
import torch
import triton
import triton.language as tl

_canonical=importlib.import_module('.mm',__package__)


@triton.jit
def _pack(B,P,N:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    inner_k=offset%64
    inner_n=(offset//64)%32
    tile_n=(offset//2048)%(N//32)
    tile_k=offset//((N//32)*2048)
    n=tile_n*32+inner_n;k=tile_k*64+inner_k
    value=tl.load(B+n.to(tl.int64)*K+k,offset<N*K,other=0.)
    tl.store(P+offset.to(tl.int64),value,offset<N*K)


@triton.jit(do_not_specialize=['M'])
def _nt_packed(A,B,C,M,N:tl.constexpr,K:tl.constexpr,
               BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr,
               GROUP_M:tl.constexpr,EVEN_M:tl.constexpr):
    pid=tl.program_id(0)
    grid_m=tl.cdiv(M,BLOCK_M);grid_n=tl.cdiv(N,BLOCK_N)
    width=GROUP_M*grid_n;group_id=pid//width
    group_size=tl.minimum(grid_m-group_id*GROUP_M,GROUP_M)
    pid_m=group_id*GROUP_M+pid%group_size
    pid_n=pid%width//group_size
    rm=pid_m*BLOCK_M+tl.arange(0,BLOCK_M)
    rn=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)
    if EVEN_M:ram=tl.max_contiguous(tl.multiple_of(rm,BLOCK_M),BLOCK_M)
    else:ram=tl.max_contiguous(tl.multiple_of(rm%M,BLOCK_M),BLOCK_M)
    rbn=tl.max_contiguous(tl.multiple_of(rn,BLOCK_N),BLOCK_N)
    rk=tl.arange(0,BLOCK_K)
    a_ptrs=A+ram[:,None]*K+rk[None,:]
    b_ptrs=B+pid_n.to(tl.int64)*2048+(rbn[None,:]%32)*64+rk[:,None]
    acc=tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32)
    for k in range(0,tl.cdiv(K,BLOCK_K)):
        a=tl.load(a_ptrs);b=tl.load(b_ptrs)
        if a.dtype!=b.dtype:
            a=a.to(C.dtype.element_ty);b=b.to(C.dtype.element_ty)
        acc=tl.dot(a,b,acc,out_dtype=tl.float32,allow_tf32=False)
        a_ptrs+=BLOCK_K
        b_ptrs+=(N//32)*2048
    c_ptrs=C+rm[:,None]*N+rn[None,:]
    result=acc.to(C.dtype.element_ty)
    if EVEN_M:tl.store(c_ptrs,result)
    else:tl.store(c_ptrs,result,(rm<M)[:,None])


def pack_into(b,packed):
    k,n=b.shape
    assert b.dtype==torch.bfloat16 and b.device.type=='cuda' and b.stride()==(1,k)
    assert n%32==0 and k%64==0 and packed.shape==(k//64,n//32,32,64)
    assert packed.dtype==b.dtype and packed.device==b.device and packed.is_contiguous()
    _pack[(triton.cdiv(n*k,1024),)](b,packed,n,k,1024,num_warps=4)
    return packed


def pack(b):
    k,n=b.shape
    result=torch.empty((k//64,n//32,32,64),device=b.device,dtype=b.dtype)
    return pack_into(b,result)


def eligible(a,b,packed,out):
    if a.ndim!=2 or b.ndim!=2 or out.ndim!=2:return False
    m,k=a.shape;kb,n=b.shape
    if not (m%32==0 and 1<=m//32<=2 and k==kb and n>=2048 and k>=4096 and n%32==0 and k%64==0):return False
    if any(t.dtype!=torch.bfloat16 or t.device!=a.device for t in (a,b,packed,out)):return False
    if a.device.type!='cuda' or not a.is_contiguous() or b.stride()!=(1,k):return False
    if out.shape!=(m,n) or not out.is_contiguous():return False
    if packed.shape!=(k//64,n//32,32,64) or not packed.is_contiguous():return False
    if max(m*k,k*n,m*n)>=2**31:return False
    pointer=out.untyped_storage().data_ptr()
    if any(pointer==t.untyped_storage().data_ptr() for t in (a,b,packed)):return False
    if _canonical._select_two_step_split_k(m,n,k) is not None:return False
    if _canonical.splitk_mm_scenario(m,n,k):return False
    return _canonical.nt_mm_scenario(a,b,out,m,n,k)


def mm_out(a,b,packed,*,out):
    if not eligible(a,b,packed,out):return _canonical.mm_out(a,b,out=out)
    m,k=a.shape;n=b.shape[1]
    _nt_packed[(triton.cdiv(m,32)*triton.cdiv(n,32),)](
        a,packed,out,m,n,k,32,32,64,8,m%32==0,
        num_warps=4,num_stages=4,pipeline='basic')
    return out


_weights={}


def _identity(tensor):
    storage=tensor.untyped_storage()
    return (tensor.device,storage._cdata,storage.nbytes(),tensor.storage_offset(),
            tuple(tensor.shape),tuple(tensor.stride()),tensor.dtype)


def register_weight(owner):
    """Only a supported model's immutable unquantized down weight is registered."""
    if owner.ndim!=2 or owner.dtype!=torch.bfloat16 or owner.device.type!='cuda':return None
    n,k=owner.shape
    if not owner.is_contiguous() or n<2048 or k<4096 or n%32 or k%64 or n*k>=2**31:return None
    key=owner.untyped_storage()._cdata
    previous=_weights.get(key)
    if previous is not None:
        if previous['owner'] is not owner or previous['identity']!=_identity(owner):return None
        pack_into(owner.T,previous['packed']);previous['ready']=True
        return previous['packed']
    packed=pack(owner.T)
    _weights[key]=dict(owner=owner,identity=_identity(owner),packed=packed,
                      b_identity=_identity(owner.T),ready=True)
    return packed


def register_model(model):
    """Called inside model-loading memory profiling, before KV budget/capture."""
    count=0;size=0
    for name,module in model.named_modules():
        if not name.endswith('.down_proj'):continue
        if type(getattr(module,'quant_method',None)).__name__!='UnquantizedLinearMethod':continue
        if getattr(module,'tp_size',1)!=1 or getattr(module,'bias',None) is not None:continue
        owner=getattr(module,'weight',None)
        if not isinstance(owner,torch.Tensor):continue
        packed=register_weight(owner)
        if packed is not None:count+=1;size+=packed.numel()*packed.element_size()
    return count,size


def begin_reload(model):
    owners={id(parameter) for parameter in model.parameters()}
    for record in _weights.values():
        if id(record['owner']) in owners:record['ready']=False


def finish_reload(model):
    owners={id(parameter) for parameter in model.parameters()}
    for record in _weights.values():
        if id(record['owner']) not in owners or record['identity']!=_identity(record['owner']):
            raise RuntimeError('Packed down weight identity changed; restart before graph reuse')
        pack_into(record['owner'].T,record['packed']);record['ready']=True


def try_registered(a,b,out):
    m=a.shape[0]
    if m%32 or not 1<=m//32<=2 or not _weights:return False
    record=_weights.get(b.untyped_storage()._cdata)
    if record is None or not record['ready'] or record['b_identity']!=_identity(b):return False
    packed=record['packed']
    if not eligible(a,b,packed,out):return False
    _nt_packed[(triton.cdiv(m,32)*triton.cdiv(b.shape[1],32),)](
        a,packed,out,m,b.shape[1],a.shape[1],32,32,64,8,True,
        num_warps=4,num_stages=4,pipeline='basic')
    return True
