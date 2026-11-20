import torch
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash as f
print(f.__module__)
nb,bs,KV,D=64,16,8,128; bf=torch.bfloat16
kv=torch.zeros(nb,2,bs,KV,D,device='cuda',dtype=bf)  # some layout; use the backend's transposed split
kv=torch.zeros(nb,KV,bs,2*D,device='cuda',dtype=bf)
kc,vc=kv.transpose(1,2).split(D,dim=-1)
k=torch.randn(37,KV,D,device='cuda',dtype=bf); v=torch.randn_like(k)
slots=torch.randperm(nb*bs,device='cuda')[:37]
one=torch.ones((),device='cuda')
f(k,v,kc,vc,slots,'auto',one,one); torch.cuda.synchronize()
rk=torch.zeros(kc.shape,device="cuda",dtype=bf); rv=torch.zeros(vc.shape,device="cuda",dtype=bf)
rk.view(-1,KV,D)[slots]=k; rv.view(-1,KV,D)[slots]=v
print('k exact', torch.equal(kc.contiguous(),rk), 'v exact', torch.equal(vc.contiguous(),rv))
