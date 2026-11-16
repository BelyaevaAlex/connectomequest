"""Bounded source-only MLP fitting, with portable numpy inference artifacts."""
import hashlib
import time
import numpy as np
from scipy import sparse

def directed_means(matrix):
    def normalize(a):
        a=a.astype('float32').tocsr()
        return sparse.diags(1/np.maximum(np.asarray(a.sum(1)).ravel(),1))@a
    return normalize(matrix),normalize(matrix.T)

class DirectionalGraphSAGE:
    """Two directed mean aggregation layers, portable inference without torch."""
    def __init__(self,mean,scale,w1,b1,w2,b2):
        for key,value in locals().copy().items():
            if key!='self':
                arr=np.array(value,dtype=np.float32,copy=True); arr.flags.writeable=False; setattr(self,key,arr)
        self.label_count=len(self.b2)

    def embeddings(self,matrix,features):
        out,inc=directed_means(matrix)
        x=(np.asarray(features,dtype=np.float32)-self.mean)/self.scale
        h=np.maximum(np.column_stack((x,out@x,inc@x))@self.w1+self.b1,0)
        return np.column_stack((h,out@h,inc@h))

    def logits(self,embeddings):
        return np.asarray(embeddings,dtype=np.float32)@self.w2+self.b2

    def save(self,path):
        with open(path,'xb') as stream: np.savez(stream,**{key:getattr(self,key) for key in ('mean','scale','w1','b1','w2','b2')})

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z: return cls(**{key:z[key] for key in ('mean','scale','w1','b1','w2','b2')})

def source_split(labels,seed=0):
    groups=[[],[],[]]
    for cell,label in sorted(labels.items()):
        if label is None: continue
        value=int.from_bytes(hashlib.sha256(f'{seed}:{cell}'.encode()).digest()[:8],'big')%100
        groups[0 if value<70 else 1 if value<85 else 2].append(int(cell))
    return tuple(tuple(group) for group in groups)

def train_graphsage(matrix,features,labels,train_rows,calibration_rows,*,label_count,seed=0,epochs=40,hidden=64,device='cpu',learning_rate=.01):
    """Full graph transductive propagation; loss only on declared train rows."""
    rows=np.asarray(train_rows,dtype=np.int64); cal=np.asarray(calibration_rows,dtype=np.int64)
    x=np.asarray(features,dtype=np.float32); y=np.asarray(labels,dtype=np.int64)
    if not 1<=epochs<=200 or not 1<=hidden<=256 or len(x)>50000 or not 1<=label_count<=20000: raise ValueError('training resource cap exceeded')
    if len(rows)==0 or len(cal)==0 or set(rows)&set(cal): raise ValueError('nonempty disjoint splits required')
    if matrix.shape!=(len(x),len(x)) or len(y)!=len(x) or not np.isfinite(x).all(): raise ValueError('invalid graph features')
    if np.any(rows<0) or np.any(rows>=len(x)) or np.any(cal<0) or np.any(cal>=len(x)) or np.any(y[rows]<0) or np.any(y[rows]>=label_count) or np.any(y[cal]<0) or np.any(y[cal]>=label_count): raise ValueError('invalid training rows/labels')
    if len(rows)*label_count>150000000: raise MemoryError('training logit cap exceeded')
    import torch
    torch.manual_seed(seed)
    mean=x[rows].mean(0); scale=np.maximum(x[rows].std(0),1e-6)
    out,inc=directed_means(matrix)
    def tensor(a):
        a=a.tocoo()
        return torch.sparse_coo_tensor(np.stack((a.row,a.col)),a.data,a.shape,device=device).coalesce()
    po,pi=tensor(out),tensor(inc)
    tx=torch.as_tensor((x-mean)/scale,device=device)
    first=torch.nn.Linear(3*x.shape[1],hidden).to(device)
    last=torch.nn.Linear(3*hidden,label_count).to(device)
    optimizer=torch.optim.Adam(list(first.parameters())+list(last.parameters()),lr=learning_rate)
    z=torch.cat((tx,torch.sparse.mm(po,tx),torch.sparse.mm(pi,tx)),dim=1)
    target=torch.as_tensor(y[rows],device=device)
    losses=[]; calibration=[]; best_loss=float('inf'); best_epoch=0; best_first=None; best_last=None
    cal_target=torch.as_tensor(y[cal],device=device)
    started=time.monotonic()
    for epoch in range(epochs):
        optimizer.zero_grad()
        h=torch.relu(first(z))
        embedding=torch.cat((h,torch.sparse.mm(po,h),torch.sparse.mm(pi,h)),dim=1)
        loss=torch.nn.functional.cross_entropy(last(embedding[rows]),target)
        loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            hc=torch.relu(first(z))
            ec=torch.cat((hc,torch.sparse.mm(po,hc),torch.sparse.mm(pi,hc)),dim=1)
            current=float(torch.nn.functional.cross_entropy(last(ec[cal]),cal_target).cpu())
        calibration.append(current)
        if current<best_loss:
            best_loss=current; best_epoch=epoch+1
            best_first={k:v.detach().cpu().clone() for k,v in first.state_dict().items()}
            best_last={k:v.detach().cpu().clone() for k,v in last.state_dict().items()}
        if time.monotonic()-started>5400: raise TimeoutError('training exceeded 90 minute cap')
    first.load_state_dict(best_first); last.load_state_dict(best_last)
    model=DirectionalGraphSAGE(mean,scale,first.weight.detach().cpu().numpy().T,first.bias.detach().cpu().numpy(),last.weight.detach().cpu().numpy().T,last.bias.detach().cpu().numpy())
    return model,dict(train_rows=len(rows),calibration_rows=len(cal),seed=seed,epochs=epochs,hidden=hidden,batch_size='full',device=device,learning_rate=learning_rate,optimizer_updates=epochs,training_loss=losses,calibration_nll_trajectory=calibration,best_epoch=best_epoch,training_wall_seconds=time.monotonic()-started,optimization_not_converged=bool(len(losses)>50 and (losses[-51]-losses[-1])/max(abs(losses[-51]),1e-12)>.01))

class StructuralMLP:
    def __init__(self,mean,scale,w1,b1,w2,b2):
        for key,value in locals().copy().items():
            if key!='self':
                arr=np.array(value,dtype=np.float32,copy=True); arr.flags.writeable=False; setattr(self,key,arr)
        self.label_count=len(self.b2)

    def logits(self,features):
        x=np.asarray(features,dtype=np.float32)
        return np.maximum((x-self.mean)/self.scale@self.w1+self.b1,0)@self.w2+self.b2

    def save(self,path):
        with open(path,'xb') as stream: np.savez(stream,**{key:getattr(self,key) for key in ('mean','scale','w1','b1','w2','b2')})

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z: return cls(**{key:z[key] for key in ('mean','scale','w1','b1','w2','b2')})

def train_mlp(x_train,y_train,x_calibration,y_calibration,*,label_count,seed=0,epochs=40,hidden=64,batch_size=256,device='cpu',learning_rate=.01):
    if not 1<=epochs<=200 or not 1<=hidden<=256 or not 1<=batch_size<=4096 or not 1<=label_count<=20000:
        raise ValueError('training resource cap exceeded')
    import torch
    x=np.asarray(x_train,dtype=np.float32); y=np.asarray(y_train,dtype=np.int64)
    xc=np.asarray(x_calibration,dtype=np.float32); yc=np.asarray(y_calibration,dtype=np.int64)
    if x.ndim!=2 or xc.ndim!=2 or x.shape[1]!=xc.shape[1] or len(x)!=len(y) or len(xc)!=len(yc) or not len(x) or not len(xc): raise ValueError('invalid aligned arrays')
    if not np.isfinite(x).all() or not np.isfinite(xc).all() or np.any(y<0) or np.any(y>=label_count) or np.any(yc<0) or np.any(yc>=label_count): raise ValueError('invalid features or labels')
    mean=x.mean(0); scale=np.maximum(x.std(0),1e-6)
    torch.manual_seed(seed)
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],hidden),torch.nn.ReLU(),torch.nn.Linear(hidden,label_count)).to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
    tx=torch.as_tensor((x-mean)/scale,device=device); ty=torch.as_tensor(y,device=device)
    rng=np.random.default_rng(seed)
    losses=[]; updates=0; calibration=[]; best_loss=float('inf'); best_state=None; best_epoch=0
    tc=torch.as_tensor((xc-mean)/scale,device=device); tc_labels=torch.as_tensor(yc,device=device)
    started=time.monotonic()
    for epoch in range(epochs):
        total=0.
        for indices in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/batch_size)))):
            optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(net(tx[indices]),ty[indices]); loss.backward(); optimizer.step()
            total+=float(loss.detach().cpu())*len(indices); updates+=1
        losses.append(total/len(x))
        with torch.no_grad(): current=float(torch.nn.functional.cross_entropy(net(tc),tc_labels).cpu())
        calibration.append(current)
        if current<best_loss:
            best_loss=current; best_epoch=epoch+1
            best_state={name:value.detach().cpu().clone() for name,value in net.state_dict().items()}
        if time.monotonic()-started>5400: raise TimeoutError('training exceeded 90 minute cap')
    net.load_state_dict(best_state)
    first,last=net[0],net[2]
    model=StructuralMLP(mean,scale,first.weight.detach().cpu().numpy().T,first.bias.detach().cpu().numpy(),last.weight.detach().cpu().numpy().T,last.bias.detach().cpu().numpy())
    return model,dict(train_rows=len(x),calibration_rows=len(xc),seed=seed,epochs=epochs,hidden=hidden,batch_size=batch_size,device=device,learning_rate=learning_rate,optimizer_updates=updates,training_loss=losses,calibration_nll_trajectory=calibration,best_epoch=best_epoch,training_wall_seconds=time.monotonic()-started)
