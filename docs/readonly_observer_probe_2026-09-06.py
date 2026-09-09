import pathlib,zipfile,pickle,io,collections,json,warnings
warnings.filterwarnings('ignore')
import numpy as np
r=pathlib.Path('/data/home/kxrgzn/lzh/MyDualTalk')
class Ref:
 def __init__(self,s,o,size,stride,*a):self.storage=s;self.offset=o;self.size=size;self.stride=stride
class U(pickle.Unpickler):
 def find_class(self,m,n):
  if m=='collections' and n=='OrderedDict':return collections.OrderedDict
  if m=='torch._utils' and n in ['_rebuild_tensor_v2','_rebuild_tensor']:return Ref
  if m=='torch' and n.endswith('Storage'):return n
  if m=='numpy' and n in ['dtype','ndarray']:return getattr(np,n)
  if m in ['numpy.core.multiarray','numpy._core.multiarray'] and n=='_reconstruct':return np.core.multiarray._reconstruct
  if m=='_codecs' and n=='encode':
   import codecs
   return codecs.encode
  raise ValueError((m,n))
 def persistent_load(self,p):return p

import sys,torch
torch.set_num_threads(1)
sys.path.insert(0,str(r/'legacy_eval_655d31a'))
from emotion_ssm.models.observation import ObservationEncoder
def read_selected(f,component,prefix):
 with zipfile.ZipFile(r/f) as z:
  header=next(n for n in z.namelist() if n.endswith('/data.pkl'))
  d=U(io.BytesIO(z.read(header))).load()
  v=d.get('models',d)
  if component:v=v[component]
  out={}
  for name,t in v.items():
   if name.startswith(prefix) and isinstance(t,Ref):
    typ={'FloatStorage':'<f4','LongStorage':'<i8','IntStorage':'<i4','HalfStorage':'<f2','ByteStorage':'u1','DoubleStorage':'<f8','BoolStorage':'?'}[t.storage[1]]
    arr=np.frombuffer(z.read(header.rsplit('/',1)[0]+'/data/'+t.storage[2]),dtype=typ)
    arr=arr[t.offset:t.offset+int(np.prod(t.size))].reshape(t.size).copy()
    out[name[len(prefix):]]=torch.from_numpy(arr)
  return out
p=r/'artifacts/features/iemocap'
ids=(p/'splits/fold_5/test_dialogues.txt').read_text().splitlines()
chosen=[ids[i] for i in np.linspace(0,len(ids)-1,8,dtype=int)]
features=[];sources=[]
for name in chosen:
 v=torch.load(p/'dialogues'/name/'audio_features.pt',map_location='cpu',weights_only=True)
 if isinstance(v,dict):
  v=next(v[k] for k in ['features','embeddings','data','audio_features','audio'] if k in v)
 ix=torch.linspace(0,len(v)-1,min(16,len(v))).long()
 features.append(v[ix].float());sources.append({'dialogue':name,'count':len(ix)})
a=torch.cat(features);n=len(a)
batch={'audio':a,'face':torch.zeros(n,35),'text':torch.zeros(n,768),'dataset_id':torch.ones(n,dtype=torch.long),'modality_mask':torch.tensor([[True,False,False]]).expand(n,-1)}
def stats(a):
 a=a.float()
 unit=torch.nn.functional.normalize(a,dim=-1)
 sim=(unit@unit.T)[~torch.eye(len(a),dtype=torch.bool)]
 s=torch.linalg.svdvals(unit-unit.mean(0))
 w=s.square();w=w/w.sum().clamp_min(1e-30)
 return {'pair_cos_mean':float(sim.mean()),'pair_cos_p05':float(torch.quantile(sim,0.05)),'direction_std_mean':float(unit.std(0).mean()),'norm_mean':float(a.norm(dim=-1).mean()),'effective_rank':float(torch.exp(-(w*w.clamp_min(1e-30).log()).sum()))}
out={'probe':'CPU, 1 thread, precomputed IEMOCAP fold5 test audio, no waveform backbone; same inputs for all encoders','samples':n,'sources':sources,'checkpoints':{}}
for label,f,c,pre in [
 ('phase_b_teacher','runs/phase_b_coupling/fold5/phase_b_best_numpy1_compat.pt','teacher',''),
 ('phase_b_encoder','runs/phase_b_coupling/fold5/phase_b_best_numpy1_compat.pt','bundle','encoder.'),
 ('conditioned_epoch3','runs/dualtalk_conditioned/fold5/dualtalk_emotion_best.pt','system','conditioner.audio_observer.observation_encoder.'),
 ('conditioned_epoch22','runs/comparison_epoch24/dualtalk_emotion_best_epoch22.pt','system','conditioner.audio_observer.observation_encoder.'),
 ('all_three_epoch23','runs/dualtalk_conditioned/all_three/dualtalk_emotion_best.pt','system','conditioner.audio_observer.observation_encoder.')]:
 model=ObservationEncoder().eval()
 model.load_state_dict(read_selected(f,c,pre),strict=True)
 with torch.no_grad():v=model(batch,torch.tensor([[True,False,False]]))
 out['checkpoints'][label]={k:stats(getattr(v,k)[:,0]) for k in ['aff','event','action','hidden']}
print(json.dumps(out))
