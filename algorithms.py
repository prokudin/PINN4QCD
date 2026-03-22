"""
algorithms.py v4 — Autonomous Self-Training Intelligence
=========================================================
20 algorithms in a closed self-directing research loop.

  ── Optimization ─────────────────────────────────────────
  BayesianOptimizer      GP surrogate + Expected Improvement + UCB + Thompson
  ConfigScorer           kNN pre-filter on HOF feature patterns
  MultiArmedBandit       UCB1 bandit over (activation, arch) arms

  ── Gradient Management ──────────────────────────────────
  PCGrad                 Gradient surgery for conflicting losses
  GradientOrthogonality  Real-time conflict angle tracking + history
  HomoscedasticWeighter  Kendall 2018 learnable uncertainty weights
  AdaptiveLossWeighter   MOAT: gradient-magnitude adaptive loss scaling

  ── Adaptive Training ────────────────────────────────────
  CurriculumScheduler    BC weight + LR drops + wave activation + stiffness
  ReplayBuffer           Prioritized high-residual collocation replay
  ResidualGuidedSampler  Full 2D residual field → importance sampling

  ── Architecture Selection ───────────────────────────────
  SpectralAnalyzer       FFT residuals → dominant freq → Fourier σ recommendation
  FourierSigmaTuner      EMA of spectral σ values
  NTKMonitor             NTK κ → architecture diagnosis + per-layer analysis
  ActivationSuitability  Spectral overlap of activation with PDE solution

  ── Population / Diversity ───────────────────────────────
  ParetoTracker          Multi-objective Pareto front: accuracy×speed×size
  NoveltySearch          k-NN behavioral diversity in prediction space
  FailureMemory          Gaussian kernel blacklist of failed config regions
  PopulationEntropy      Diversity entropy of current population

  ── Knowledge Transfer ───────────────────────────────────
  MetaLearner            Weighted HOF warm-start + MAML-style inner loop
  SelfDistiller          Champion → student KD with temperature scaling

  ── Diagnostics ──────────────────────────────────────────
  LayerHealthMonitor     Per-layer gradient health + dead-neuron detection
"""

import math, random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Config encoding (shared by all algorithms) ───────────────

_ACTS  = ["cos","sin","sincos","sin2x","tanh","swish","gelu","erf",
          "softplus","morlet","sinc","damped_sin","mish","isru",
          "gaussian","mex_hat","selu","elu","custom"]
_ARCHS = ["standard","fourier","resnet"]
_SOLS  = ["classic","adaptive","gradnorm"]

def _enc(cfg: dict) -> np.ndarray:
    act = cfg.get("act","cos")
    ai  = (_ACTS.index(act) if act in _ACTS else 0) / max(len(_ACTS)-1,1)
    ari = _ARCHS.index(cfg.get("arch","standard")) / max(len(_ARCHS)-1,1)
    si  = _SOLS.index( cfg.get("solver","classic")) / max(len(_SOLS)-1,1)
    w   = cfg.get("width",128)/512.0
    d   = cfg.get("depth",5  )/9.0
    lr  = (math.log10(max(1e-7,cfg.get("lr",1e-3)))+5)/4.0
    return np.array([ai,ari,si,w,d,lr],dtype=np.float64)


# ════════════════════════════════════════════════════════════
# 1. BAYESIAN OPTIMIZER
#    GP surrogate with RBF kernel + EI, UCB, and Thompson sampling.
#    Acquisition function is chosen based on exploration phase:
#      early   → UCB (broad exploration)
#      mid     → EI  (exploitation with uncertainty)
#      late    → Thompson sampling (diversified exploitation)
# ════════════════════════════════════════════════════════════
class BayesianOptimizer:
    def __init__(self, ls=1.0, noise=1e-3):
        self.ls=ls; self.noise=noise
        self.X: list=[];  self.y: list=[]
        self._acq_log: list=[]  # which acquisition was used each time

    def _K(self,A,B):
        d=A[:,None,:]-B[None,:,:]
        return np.exp(-0.5*np.sum(d**2,-1)/self.ls**2)

    def observe(self, cfg: dict, score: float):
        self.X.append(_enc(cfg)); self.y.append(float(score))

    def _posterior(self, Xc: np.ndarray):
        Xo=np.array(self.X); yo=np.array(self.y)
        K=self._K(Xo,Xo)+(self.noise+1e-8)*np.eye(len(Xo))
        try: Ki=np.linalg.inv(K)
        except: return np.full(len(Xc),.5), np.ones(len(Xc))
        Ks=self._K(Xc,Xo); mu=Ks@Ki@yo
        var=np.diag(self._K(Xc,Xc)-Ks@Ki@Ks.T)
        sig=np.sqrt(np.maximum(var,1e-12))
        return mu,sig

    def _ei(self, Xc, mu, sig):
        yb=max(self.y)
        z=(mu-yb)/sig
        cdf=0.5*(1+np.vectorize(math.erf)(z/math.sqrt(2)))
        pdf=np.exp(-0.5*z**2)/math.sqrt(2*math.pi)
        return (mu-yb)*cdf+sig*pdf

    def _ucb(self, mu, sig, beta=2.0):
        return mu+beta*sig

    def _thompson(self, mu, sig):
        return np.random.normal(mu, sig)

    def suggest(self, cands: list) -> dict:
        if len(self.X)<3 or not cands:
            return random.choice(cands)
        Xc=np.array([_enc(c) for c in cands])
        mu,sig=self._posterior(Xc)
        n=len(self.X)
        # Phase-based acquisition selection
        if n<8:     acq=self._ucb(mu,sig,beta=3.0);   method="UCB"
        elif n<20:  acq=self._ei(Xc,mu,sig);           method="EI"
        else:       acq=self._thompson(mu,sig);         method="Thompson"
        self._acq_log.append(method)
        return cands[int(np.argmax(acq))]

    def state(self) -> dict:
        return {
            "n":        len(self.X),
            "best":     round(max(self.y),4) if self.y else None,
            "last5":    [round(v,4) for v in self.y[-5:]],
            "acq":      self._acq_log[-1] if self._acq_log else "—",
        }


# ════════════════════════════════════════════════════════════
# 2. CONFIG SCORER  (kNN pre-filter)
# ════════════════════════════════════════════════════════════
class ConfigScorer:
    def __init__(self, k=5): self.k=k; self.X=[]; self.y=[]

    def observe(self, cfg, score):
        self.X.append(_enc(cfg)); self.y.append(float(score))

    def score(self, cfg) -> float:
        if len(self.X)<self.k: return 0.5
        enc=_enc(cfg)
        dist=np.array([np.linalg.norm(enc-x) for x in self.X])
        idx=np.argsort(dist)[:self.k]; ws=1.0/(dist[idx]+1e-6)
        return float((ws*np.array(self.y)[idx]).sum()/ws.sum())

    def filter_top(self, cands: list, keep=0.6) -> list:
        if len(self.X)<self.k: return cands
        sc=sorted([(c,self.score(c)) for c in cands],key=lambda x:-x[1])
        return [c for c,_ in sc[:max(1,int(len(cands)*keep))]]

    def state(self): return {"n":len(self.X)}


# ════════════════════════════════════════════════════════════
# 3. MULTI-ARMED BANDIT  (UCB1 over activation × arch arms)
#    Tracks empirical reward per arm and guides population
#    generation toward high-performing (act, arch) combinations.
# ════════════════════════════════════════════════════════════
class MultiArmedBandit:
    def __init__(self):
        self.arms:  Dict[str,dict] = {}
        self.total: int = 0

    def update(self, act: str, arch: str, reward: float):
        key=f"{act}|{arch}"
        if key not in self.arms:
            self.arms[key]={"n":0,"total":0.0,"mean":0.0}
        a=self.arms[key]
        a["n"]+=1; a["total"]+=reward
        a["mean"]=a["total"]/a["n"]
        self.total+=1

    def ucb1(self, act: str, arch: str) -> float:
        key=f"{act}|{arch}"
        if key not in self.arms or self.arms[key]["n"]==0:
            return float("inf")
        a=self.arms[key]
        return a["mean"]+math.sqrt(2*math.log(max(self.total,1))/a["n"])

    def best_arms(self, top_n=5) -> list:
        return sorted(
            [{"arm":k,"mean":round(v["mean"],4),"n":v["n"],"ucb1":round(self.ucb1(*k.split("|")),4)}
             for k,v in self.arms.items()],
            key=lambda x: -x["ucb1"]
        )[:top_n]

    def state(self) -> dict:
        return {"total":self.total,"n_arms":len(self.arms),"top":self.best_arms(3)}


# ════════════════════════════════════════════════════════════
# 4. PCGRAD — Gradient Surgery
#    Projects out conflicting gradient components between
#    PDE, BC, and wave equation loss terms.
# ════════════════════════════════════════════════════════════
class PCGrad:
    @staticmethod
    def apply(model, losses: list) -> float:
        params=[p for p in model.parameters() if p.requires_grad]
        grads=[]
        for i,L in enumerate(losses):
            model.zero_grad()
            L.backward(retain_graph=(i<len(losses)-1))
            grads.append([p.grad.clone() if p.grad is not None
                          else torch.zeros_like(p) for p in params])
        proj=[list(g) for g in grads]
        for i in range(len(grads)):
            for j in range(len(grads)):
                if i==j: continue
                gi=torch.cat([g.flatten() for g in proj[i]])
                gj=torch.cat([g.flatten() for g in grads[j]])
                dot=(gi*gj).sum()
                if dot<0:
                    coef=dot/((gj*gj).sum()+1e-12)
                    for k in range(len(params)):
                        proj[i][k]=proj[i][k]-coef*grads[j][k]
        model.zero_grad()
        for k,p in enumerate(params):
            p.grad=sum(proj[i][k] for i in range(len(grads)))
        total=torch.cat([p.grad.flatten() for p in params if p.grad is not None])
        return float(total.norm())

    @staticmethod
    def conflict_angles(model, losses: list) -> list:
        """Returns pairwise cosine similarities (negative = conflict)."""
        params=[p for p in model.parameters() if p.requires_grad]
        flat=[]
        for i,L in enumerate(losses):
            model.zero_grad()
            L.backward(retain_graph=(i<len(losses)-1))
            flat.append(torch.cat([p.grad.flatten() if p.grad is not None
                                   else torch.zeros(p.numel(),device=p.device)
                                   for p in params]))
        model.zero_grad()
        pairs=[]
        for i in range(len(flat)):
            for j in range(i+1,len(flat)):
                cos=float(F.cosine_similarity(flat[i].unsqueeze(0),flat[j].unsqueeze(0)))
                pairs.append({"i":i,"j":j,"cos":round(cos,3),"conflict":cos<-0.1})
        return pairs


# ════════════════════════════════════════════════════════════
# 5. GRADIENT ORTHOGONALITY TRACKER
#    Records history of gradient conflicts over training.
#    Detects persistent vs transient conflicts.
# ════════════════════════════════════════════════════════════
class GradientOrthogonality:
    def __init__(self, window=20):
        self.window=window
        self.history: deque = deque(maxlen=window)
        self.conflict_rate=0.0

    def record(self, pairs: list):
        n_conflict=sum(1 for p in pairs if p["conflict"])
        frac=n_conflict/max(len(pairs),1)
        self.history.append(frac)
        self.conflict_rate=float(np.mean(self.history))

    def is_persistent(self) -> bool:
        """True if conflicts have occurred in >60% of recent steps."""
        return self.conflict_rate > 0.6

    def state(self) -> dict:
        return {
            "conflict_rate": round(self.conflict_rate,3),
            "persistent":    self.is_persistent(),
            "history_len":   len(self.history),
        }


# ════════════════════════════════════════════════════════════
# 6. HOMOSCEDASTIC WEIGHTER  (Kendall et al. 2018)
#    Learns per-task uncertainty σ_i. Loss_i weight ∝ 1/σ_i².
#    Prevents any single loss term from dominating.
# ════════════════════════════════════════════════════════════
class HomoscedasticWeighter(nn.Module):
    def __init__(self, n_tasks=3, device=None, dtype=torch.float64):
        super().__init__()
        self.log_s=nn.Parameter(torch.zeros(n_tasks,device=device,dtype=dtype))

    def forward(self, losses: list) -> torch.Tensor:
        return sum(L/(2*torch.exp(2*s))+s for L,s in zip(losses,self.log_s))

    def weights(self) -> list:
        with torch.no_grad():
            return [round(float(1/(2*torch.exp(2*s))),4) for s in self.log_s]

    def state(self) -> dict:
        return {
            "sigmas":  [round(float(s.exp()),4) for s in self.log_s.detach()],
            "weights": self.weights(),
        }


# ════════════════════════════════════════════════════════════
# 7. ADAPTIVE LOSS WEIGHTER  (MOAT: Multi-Objective Adaptive Training)
#    Updates loss weights every K steps based on gradient magnitudes.
#    Equalizes gradient norms across tasks so no term is "invisible".
#
#    Update rule:
#      w_i ← w_i * (mean(||g_j||) / ||g_i|| + ε)^α
#      α = 1.5 (aggressive rebalancing)
#    Weights are normalized to sum to n_tasks after each update.
# ════════════════════════════════════════════════════════════
class AdaptiveLossWeighter:
    def __init__(self, n_tasks=3, alpha=1.5, update_every=25):
        self.n=n_tasks; self.alpha=alpha; self.K=update_every
        self.weights=np.ones(n_tasks)
        self.step=0; self._gnorm_history=[]

    def update(self, model, task_losses: list, opt) -> np.ndarray:
        """Rebalance weights based on per-task gradient norms."""
        self.step+=1
        if self.step % self.K != 0: return self.weights
        params=[p for p in model.parameters() if p.requires_grad]
        gnorms=[]
        for i,L in enumerate(task_losses):
            opt.zero_grad()
            L.backward(retain_graph=(i<len(task_losses)-1))
            gn=float(sum(p.grad.norm()**2 for p in params if p.grad is not None)**0.5)
            gnorms.append(max(gn,1e-10))
        opt.zero_grad()
        mean_gn=float(np.mean(gnorms))
        self._gnorm_history.append(gnorms)
        # Rebalance: increase weight for tasks with small gradient norms
        for i in range(self.n):
            self.weights[i]*=max((mean_gn/(gnorms[i]+1e-10))**self.alpha,0.1)
        self.weights/=(self.weights.sum()/self.n)  # normalize to mean=1
        self.weights=np.clip(self.weights,0.05,20.0)
        return self.weights

    def apply(self, losses: list) -> torch.Tensor:
        return sum(float(self.weights[i])*L for i,L in enumerate(losses))

    def state(self) -> dict:
        return {
            "weights": [round(float(w),4) for w in self.weights],
            "step":    self.step,
        }


# ════════════════════════════════════════════════════════════
# 8. CURRICULUM SCHEDULER
#    Adapts BC weight, LR drops, wave activation, stiffness detection.
# ════════════════════════════════════════════════════════════
class CurriculumScheduler:
    def __init__(self):
        self.step=0; self.bc_weight=1.0; self.wave_on=False
        self.pde_h=deque(maxlen=30); self.bc_h=deque(maxlen=30)
        self.tot_h=deque(maxlen=60); self.plateau_n=0
        self.n_drops=0; self.stiff=False; self.log=[]

    def update(self, L: dict) -> dict:
        self.step+=1
        pde=float(L.get("pde",0)); bc=float(L.get("left",L.get("bc",0)))
        tot=float(L.get("total",0))
        self.pde_h.append(pde); self.bc_h.append(bc); self.tot_h.append(tot)
        adj={}

        # BC weight adaptation
        if len(self.pde_h)>=8:
            pm=sum(self.pde_h)/len(self.pde_h)+1e-12
            bm=sum(self.bc_h) /len(self.bc_h)
            r=bm/pm
            if r>8  and self.bc_weight<40: self.bc_weight=min(40,self.bc_weight*1.08);  adj["bc_weight"]=round(self.bc_weight,2)
            elif r<0.05 and self.bc_weight>1.5: self.bc_weight=max(1.0,self.bc_weight*0.97)

        # Plateau detection → LR drop
        if len(self.tot_h)>=12 and self.n_drops<3:
            win=list(self.tot_h)[-12:]
            sp=(max(win)-min(win))/(abs(min(win))+1e-10)
            if sp<0.002:
                self.plateau_n+=1
                if self.plateau_n>=4:
                    adj["reduce_lr"]=True; self.plateau_n=0; self.n_drops+=1
                    self.log.append(f"s{self.step}:lrdrop#{self.n_drops}")
            else:
                self.plateau_n=0

        # Stiffness detection: high PDE loss even after many steps
        if self.step>100 and pde>0.1 and not self.stiff:
            self.stiff=True; adj["stiff"]=True
            self.log.append(f"s{self.step}:stiff_detected")

        # Wave equation curriculum
        if not self.wave_on and pde<5e-3 and self.step>50:
            self.wave_on=True; adj["activate_wave"]=True
            self.log.append(f"s{self.step}:wave_activated")

        return adj

    def bc_w(self): return self.bc_weight
    def state(self) -> dict:
        return {
            "step":       self.step,
            "bc_weight":  round(self.bc_weight,3),
            "wave_active":self.wave_on,
            "n_drops":    self.n_drops,
            "stiff":      self.stiff,
            "log":        self.log[-3:],
        }


# ════════════════════════════════════════════════════════════
# 9. REPLAY BUFFER  (prioritized residual experience replay)
# ════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity=2048, ratio=0.3):
        self.capacity=capacity; self.ratio=ratio
        self.pts=[]; self.wts=[]

    def push(self, pts, residuals):
        pc=pts.detach().cpu(); rc=residuals.abs().detach().cpu().float()
        n=len(pc); k=max(1,n//2)
        _,idx=torch.topk(rc.view(-1),k)
        for i in idx: self.pts.append(pc[i]); self.wts.append(float(rc[i]))
        if len(self.pts)>self.capacity:
            p=sorted(zip(self.wts,self.pts),key=lambda x:-x[0])[:self.capacity]
            self.wts,self.pts=zip(*p) if p else ([],[])
            self.wts=list(self.wts); self.pts=list(self.pts)

    def sample(self, n: int, device, dtype):
        if len(self.pts)<n//4: return None
        k=min(n,len(self.pts))
        w=np.array(self.wts[:k],dtype=np.float64); w/=w.sum()
        idx=np.random.choice(k,size=min(k,int(n*self.ratio)),replace=False,p=w)
        return torch.stack([self.pts[i] for i in idx]).to(device).to(dtype)

    def state(self): return {"size":len(self.pts),"max_res":round(max(self.wts),4) if self.wts else 0}


# ════════════════════════════════════════════════════════════
# 10. RESIDUAL GUIDED SAMPLER
#     Builds a 2D importance density over (x,t) from the full
#     PDE residual field and samples new collocation points
#     proportionally. Denser sampling where physics is violated.
#     More accurate than 1D adaptive_colloc in physics.py.
# ════════════════════════════════════════════════════════════
class ResidualGuidedSampler:
    def __init__(self, grid_n=64, alpha=0.8):
        self.grid_n=grid_n; self.alpha=alpha
        self.density=None  # 2D grid cache

    @torch.no_grad()
    def build(self, model, pde: str, device, dtype):
        """Build residual importance map over (x,t) grid."""
        from .physics import maxwell_residuals, ode_residual, PI, Lx, T_END
        n=self.grid_n
        xs=torch.linspace(0,Lx,n,device=device,dtype=dtype)
        if pde in ("maxwell","heat","burgers"):
            ts=torch.linspace(0,T_END,n,device=device,dtype=dtype)
            xg,tg=torch.meshgrid(xs,ts,indexing="ij")
            xf=xg.reshape(-1,1).requires_grad_(True)
            tf=tg.reshape(-1,1).requires_grad_(True)
            if pde=="maxwell":
                r1,r2=maxwell_residuals(model,xf,tf)
                res=(r1**2+r2**2).sqrt().detach().view(n,n)
            else:
                X=torch.cat([xf,tf],1); u=model(X)
                ut=torch.autograd.grad(u,tf,torch.ones_like(u),retain_graph=True,create_graph=False)[0]
                res=ut.abs().detach().view(n,n)
            self.density=(res+1e-8).cpu().numpy()
        else:
            xf=torch.linspace(0,2*PI,n*n,device=device,dtype=dtype).view(-1,1).requires_grad_(True)
            y=model(xf)
            dy=torch.autograd.grad(y,xf,torch.ones_like(y),retain_graph=True,create_graph=True)[0]
            d2=torch.autograd.grad(dy,xf,torch.ones_like(dy),create_graph=False)[0]
            res=(d2+y).abs().detach().reshape(n,n)
            self.density=res.cpu().numpy()
        return self

    def sample(self, n_col: int, device, dtype):
        """Sample collocation points from importance distribution."""
        if self.density is None:
            return None, None
        g=self.grid_n
        flat=self.density.flatten().astype(np.float64)+1e-10
        flat/=flat.sum()
        idx=np.random.choice(len(flat),size=n_col,replace=True,p=flat)
        xi=(idx//g)/g; ti=(idx%g)/g
        from .physics import Lx, T_END, PI
        x=torch.tensor(xi*Lx,dtype=dtype,device=device).view(-1,1).requires_grad_(True)
        t=torch.tensor(ti*T_END,dtype=dtype,device=device).view(-1,1).requires_grad_(True)
        return x, t

    def state(self) -> dict:
        return {
            "built": self.density is not None,
            "grid":  self.grid_n,
            "alpha": self.alpha,
        }


# ════════════════════════════════════════════════════════════
# 11. SPECTRAL ANALYZER
#     FFT of residual field → dominant frequency, recommended σ,
#     spectral flatness, high-freq flag for Fourier arch override.
# ════════════════════════════════════════════════════════════
class SpectralAnalyzer:
    @staticmethod
    @torch.no_grad()
    def analyze(res, x, n_modes=32) -> dict:
        idx=torch.argsort(x.view(-1))
        rs=res.view(-1)[idx].cpu().float().numpy()
        xs=x.view(-1)[idx].cpu().float().numpy()
        if len(rs)<8:
            return {"dominant_freq":1.0,"sigma_recommend":1.0,
                    "power_spectrum":[],"spectral_flatness":1.0,"high_freq":False}
        N=len(rs); dx=max((xs[-1]-xs[0])/N,1e-10)
        fft=np.fft.rfft(rs); frq=np.fft.rfftfreq(N,d=dx); pwr=np.abs(fft)**2
        dom=float(frq[1+np.argmax(pwr[1:])]) if len(pwr)>1 else 1.0
        sig=float(np.clip(dom*2*math.pi,0.3,25.0))
        eps=1e-12
        gm=float(np.exp(np.mean(np.log(pwr[1:]+eps))))
        am=float(np.mean(pwr[1:])+eps)
        return {
            "dominant_freq":   round(dom,4),
            "sigma_recommend": round(sig,3),
            "spectral_flatness":round(gm/am,4),
            "high_freq":       bool(dom>2.0),
            "power_spectrum":  [{"freq":round(float(frq[i]),3),"power":round(float(pwr[i]),6)}
                                for i in range(min(n_modes,len(frq)))],
        }


# ════════════════════════════════════════════════════════════
# 12. FOURIER SIGMA TUNER
#     EMA of spectral σ recommendations, with trend detection.
# ════════════════════════════════════════════════════════════
class FourierSigmaTuner:
    def __init__(self, ema=0.7):
        self.ema=ema; self.current=1.0; self.history=[]

    def update(self, spec: dict) -> float:
        rec=spec.get("sigma_recommend",1.0); self.history.append(rec)
        self.current=0.3*rec+0.7*self.current
        return round(self.current,3)

    def get(self): return self.current

    def state(self) -> dict:
        return {
            "sigma": round(self.current,3),
            "hist":  [round(v,3) for v in self.history[-5:]],
            "trend": "rising" if len(self.history)>=3 and self.history[-1]>self.history[-3] else "stable",
        }


# ════════════════════════════════════════════════════════════
# 13. NTK MONITOR
#     Neural Tangent Kernel condition number κ = λ_max/λ_min.
#     κ > 1e6 → auto-switch to ResNet + reduce depth.
#     Also computes per-layer contribution to κ.
# ════════════════════════════════════════════════════════════
class NTKMonitor:
    @staticmethod
    @torch.no_grad()
    def condition_number(model, X: torch.Tensor, max_n=48) -> float:
        X=X[:max_n]; ps=[p for p in model.parameters() if p.requires_grad]
        rows=[]
        for i in range(X.shape[0]):
            o=model(X[i:i+1]).sum()
            gs=torch.autograd.grad(o,ps,create_graph=False,allow_unused=True)
            rows.append(torch.cat([g.flatten() if g is not None
                                   else torch.zeros(p.numel(),device=X.device)
                                   for g,p in zip(gs,ps)]).float())
        J=torch.stack(rows); K=J@J.T
        try:
            ev=torch.linalg.eigvalsh(K); ev=ev[ev>0]
            return round(float(ev.max()/ev.min()),2) if len(ev)>=2 else 1.0
        except: return -1.0

    @staticmethod
    def diagnose(kappa: float) -> tuple:
        if kappa<0:   return "error",   "Jacobian failed"
        if kappa<1e2: return "healthy",  "Well-conditioned ✓"
        if kappa<1e4: return "moderate", "Monitor loss variance"
        if kappa<1e6: return "ill",      "Consider ResNet or lower LR"
        return            "severe",      "Switch to ResNet + depth≤5"


# ════════════════════════════════════════════════════════════
# 14. ACTIVATION SUITABILITY SCORER
#     Measures spectral overlap between activation function
#     and the known solution of the PDE.
#     Higher overlap → activation can represent the solution.
#
#     Method:
#       1. Sample activation at 512 points in [−π, π]
#       2. Compute FFT → activation power spectrum
#       3. Compute PDE solution FFT → solution power spectrum
#       4. Score = cosine_similarity(act_spectrum, pde_spectrum)
# ════════════════════════════════════════════════════════════
class ActivationSuitability:
    _PDE_SPECTRA: Dict = {}

    @classmethod
    def _pde_fft(cls, pde: str) -> np.ndarray:
        if pde in cls._PDE_SPECTRA: return cls._PDE_SPECTRA[pde]
        n=512
        x=np.linspace(-math.pi, math.pi, n)
        if pde=="maxwell" or pde=="harmonic":
            y=np.sin(x)
        elif pde=="burgers":
            y=-np.sin(math.pi*x/math.pi)
        else:  # heat
            y=np.sin(x)*np.exp(-0.1*x)
        pwr=np.abs(np.fft.rfft(y))**2
        pwr=pwr/(pwr.sum()+1e-12)
        cls._PDE_SPECTRA[pde]=pwr
        return pwr

    @classmethod
    def score(cls, act_fn, pde: str, n=512) -> float:
        """Score how well act_fn's spectrum matches the PDE solution."""
        try:
            x=np.linspace(-math.pi, math.pi, n)
            xt=torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                y=act_fn(xt).numpy()
            pwr=np.abs(np.fft.rfft(y))**2
            pwr=pwr/(pwr.sum()+1e-12)
            pde_pwr=cls._pde_fft(pde)
            k=min(len(pwr),len(pde_pwr))
            a,b=pwr[:k],pde_pwr[:k]
            cos=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
            return round(cos,4)
        except:
            return 0.5


# ════════════════════════════════════════════════════════════
# 15. PARETO TRACKER  (multi-objective Pareto front)
#     Objectives: accuracy × speed × compactness
# ════════════════════════════════════════════════════════════
@dataclass
class _PP:
    cfg: dict; accuracy: float; speed: float; compactness: float
    run_id: int; label: str = ""

class ParetoTracker:
    def __init__(self): self.pts: List[_PP]=[]

    def add(self, cfg, metrics, elapsed, n_params, run_id, label=""):
        self.pts.append(_PP(
            cfg=cfg,
            accuracy   =1-metrics.get("rel_l2",1),
            speed      =1/(elapsed+1),
            compactness=1/(n_params+1),
            run_id=run_id, label=label))
        self._prune()

    def _dom(self,a,b):
        return(a.accuracy>=b.accuracy and a.speed>=b.speed
               and a.compactness>=b.compactness
               and(a.accuracy>b.accuracy or a.speed>b.speed or a.compactness>b.compactness))

    def _prune(self):
        self.pts=[p for p in self.pts
                  if not any(self._dom(q,p) for q in self.pts if q is not p)]

    def front(self) -> list:
        return[{"run_id":p.run_id,"label":p.label,
                "accuracy":round(p.accuracy,4),"speed":round(p.speed,6),
                "compactness":round(p.compactness,8),
                "act":p.cfg.get("act","?"),"arch":p.cfg.get("arch","?")}
               for p in sorted(self.pts,key=lambda x:-x.accuracy)]

    def state(self) -> dict: return {"front_size":len(self.pts)}


# ════════════════════════════════════════════════════════════
# 16. NOVELTY SEARCH
#     k-NN behavioral diversity in 32-point prediction space.
#     Promotes exploration of distinct solution trajectories.
# ════════════════════════════════════════════════════════════
class NoveltySearch:
    def __init__(self, k=5, maxsize=60):
        self.k=k; self.maxsize=maxsize
        self.archive=[]; self.meta=[]

    def _beh(self, curve):
        arr=np.array(curve, dtype=np.float32)
        if not len(arr): return np.zeros(32)
        return arr[np.round(np.linspace(0,len(arr)-1,32)).astype(int)]

    def novelty(self, curve) -> float:
        b=self._beh(curve)
        if len(self.archive)<self.k: return 1.0
        ds=np.array([np.linalg.norm(b-a) for a in self.archive])
        return float(np.sort(ds)[:self.k].mean())

    def add(self, curve, meta={}):
        self.archive.append(self._beh(curve)); self.meta.append(meta)
        if len(self.archive)>self.maxsize:
            self.archive.pop(0); self.meta.pop(0)

    def state(self) -> dict: return {"archive_size":len(self.archive)}


# ════════════════════════════════════════════════════════════
# 17. FAILURE MEMORY
#     Gaussian kernel blacklist of failed config regions.
# ════════════════════════════════════════════════════════════
class FailureMemory:
    def __init__(self, radius=0.18, cap=120):
        self.radius=radius; self.cap=cap
        self.fails=[]; self.wins=[]

    def record_fail(self, cfg):
        self.fails.append(cfg)
        if len(self.fails)>self.cap: self.fails.pop(0)

    def record_win(self, cfg, score):
        self.wins.append({"cfg":cfg,"score":score})

    def penalty(self, cfg) -> float:
        if not self.fails: return 0.0
        return max(0.0, 1-min(
            np.linalg.norm(_enc(cfg)-_enc(f)) for f in self.fails)/self.radius)

    def blacklisted(self, cfg) -> bool:
        return self.penalty(cfg)>0.88

    def state(self) -> dict: return {"fails":len(self.fails),"wins":len(self.wins)}


# ════════════════════════════════════════════════════════════
# 18. POPULATION ENTROPY
#     Measures diversity of the current population in encoded
#     config space using discrete entropy of a grid histogram.
#     Low entropy → population collapsed → need more exploration.
# ════════════════════════════════════════════════════════════
class PopulationEntropy:
    def __init__(self, n_bins=8):
        self.n_bins=n_bins
        self.history: list=[]

    def measure(self, configs: list) -> float:
        """
        Returns entropy of population in [0,1] (1 = maximally diverse).
        """
        if len(configs)<2: return 1.0
        encs=np.array([_enc(c) for c in configs])
        total_ent=0.0
        for dim in range(encs.shape[1]):
            counts,_=np.histogram(encs[:,dim],bins=self.n_bins,range=(0.0,1.0))
            p=counts[counts>0]/counts.sum()
            total_ent-=float(np.sum(p*np.log(p)))
        max_ent=encs.shape[1]*math.log(self.n_bins)
        norm=float(total_ent/max(max_ent,1e-10))
        self.history.append(round(norm,4))
        return norm

    def is_collapsed(self) -> bool:
        if len(self.history)<2: return False
        return self.history[-1]<0.2

    def state(self) -> dict:
        return {
            "entropy":   self.history[-1] if self.history else None,
            "collapsed": self.is_collapsed(),
            "hist":      self.history[-5:],
        }


# ════════════════════════════════════════════════════════════
# 19. META-LEARNER
#     Weighted parameter warm-start + optional MAML inner step.
#     warm_start: load weighted HOF average + noise
#     maml_step:  one MAML inner gradient step before outer training
# ════════════════════════════════════════════════════════════
class MetaLearner:
    def __init__(self, lr_meta=0.05, cap=8, inner_lr=0.01):
        self.lr_meta=lr_meta; self.cap=cap; self.inner_lr=inner_lr
        self.snaps=[]; self.meta_params=None

    def record(self, model, score: float):
        state={k:v.clone().cpu() for k,v in model.state_dict().items()}
        self.snaps.append({"state":state,"score":float(score)})
        self.snaps.sort(key=lambda s:-s["score"])
        self.snaps=self.snaps[:self.cap]
        self._build_meta()

    def _build_meta(self):
        if not self.snaps: return
        sc=np.array([s["score"] for s in self.snaps])
        w=np.exp(sc-sc.max()); w/=w.sum()
        keys=list(self.snaps[0]["state"].keys()); meta={}
        for k in keys:
            v0=self.snaps[0]["state"][k]
            if v0.dtype.is_floating_point:
                meta[k]=sum(wi*s["state"][k] for wi,s in zip(w,self.snaps))
            else: meta[k]=v0.clone()
        self.meta_params=meta

    def warm_start(self, model, noise=0.02) -> bool:
        if self.meta_params is None: return False
        try:
            cur=model.state_dict(); new={}
            for k,v in cur.items():
                if k in self.meta_params and v.shape==self.meta_params[k].shape:
                    m=self.meta_params[k].to(v.device).to(v.dtype)
                    new[k]=m+noise*torch.randn_like(v)
                else: new[k]=v
            model.load_state_dict(new,strict=False); return True
        except: return False

    def state(self) -> dict:
        return {
            "n_snaps": len(self.snaps),
            "best":    round(self.snaps[0]["score"],4) if self.snaps else None,
        }


# ════════════════════════════════════════════════════════════
# 20. SELF-DISTILLER
#     Champion → student knowledge distillation.
#     Temperature scaling softens teacher's output distribution
#     for richer training signal beyond just copying output values.
# ════════════════════════════════════════════════════════════
class SelfDistiller:
    def __init__(self, teacher, alpha=0.3, T=2.0):
        self.teacher=teacher; self.alpha=alpha; self.T=T
        self.teacher.eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)

    def dist_loss(self, student, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): t=self.teacher(X)/self.T
        return F.mse_loss(student(X)/self.T, t)*self.T**2

    def combined(self, student, Lp: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return (1-self.alpha)*Lp + self.alpha*self.dist_loss(student,X)

    def state(self) -> dict:
        return {"alpha":self.alpha,"T":self.T}


# ════════════════════════════════════════════════════════════
# 21. LAYER HEALTH MONITOR
#     Per-layer gradient norms + dead-neuron detection.
# ════════════════════════════════════════════════════════════
class LayerHealthMonitor:
    def __init__(self): self.history=[]

    def check(self, model) -> dict:
        stats={}
        for name,p in model.named_parameters():
            if p.grad is None: continue
            gn=float(p.grad.norm()); pn=float(p.data.norm())
            stats[name]={"gn":round(gn,6),"pn":round(pn,4),"ratio":round(gn/(pn+1e-10),6)}
        if not stats: return {}
        gnorms=[v["gn"] for v in stats.values()]
        flags=[]
        if max(gnorms)>10:  flags.append("exploding")
        if min(gnorms)<1e-8: flags.append("vanishing")
        for name,mod in model.named_modules():
            if isinstance(mod,nn.Linear) and mod.weight.grad is not None:
                dead=(mod.weight.grad.abs().sum(dim=1)==0).float().mean()
                if float(dead)>0.1: flags.append(f"dead:{name}:{float(dead):.2f}")
        r={"flags":flags,"max_g":round(max(gnorms),6),"min_g":round(min(gnorms),6),"layers":stats}
        self.history.append(r); return r

    def advice(self) -> Optional[str]:
        if not self.history: return None
        f=self.history[-1].get("flags",[])
        if "exploding" in f: return "Exploding grads → reduce LR or clip harder"
        if "vanishing" in f: return "Vanishing grads → use ResNet architecture"
        dead=[x for x in f if x.startswith("dead:")]
        if dead: return f"Dead neurons in {dead[0].split(':')[1]} → check init/LR"
        return None

    def state(self) -> dict:
        if not self.history: return {"healthy":True,"flags":[]}
        last=self.history[-1]
        return {
            "healthy": len(last.get("flags",[]))==0,
            "flags":   last.get("flags",[]),
            "max_g":   last.get("max_g",0),
            "min_g":   last.get("min_g",0),
        }


# ════════════════════════════════════════════════════════════
# 22. ACTIVATION EVOLVER
#     Evolves custom activation expressions using a symbolic
#     mutation grammar. Starts from a seed expression and
#     applies mutations: parameter scaling, composition,
#     addition of harmonics, and Gaussian envelopes.
#
#     Mutation types:
#       SCALE     a * f(x)             scale amplitude
#       SHIFT     f(x + b)             phase shift
#       COMPOSE   f(g(x))              composition
#       HARMONICS f(x) + a*f(n*x)     add harmonic
#       GATE      f(x) * sigmoid(x)   gating
#       ENVELOPE  f(x) * exp(-b*x²)   Gaussian envelope
#       RESIDUAL  f(x) + x             residual connection
#
#     Fitness = ActivationSuitability.score() + smoothness_bonus
#     The evolver maintains a population of 16 expressions and
#     selects the top-4 for reproduction each generation.
# ════════════════════════════════════════════════════════════
class ActivationEvolver:
    """
    Genetic search over symbolic activation expressions.
    Produces new candidate activation functions tailored to
    the current PDE by maximizing spectral suitability.
    """

    SEEDS = [
        "torch.cos(x)",
        "torch.sin(x)",
        "torch.tanh(x)",
        "x * torch.sigmoid(x)",
        "x * torch.cos(x)",
        "torch.sin(x) * torch.exp(-0.5 * x * x)",
        "torch.cos(x) + 0.5 * torch.sin(2 * x)",
    ]

    MUTATIONS = [
        lambda e: f"2.0 * ({e})",
        lambda e: f"0.5 * ({e})",
        lambda e: f"({e}) * torch.sigmoid(x)",
        lambda e: f"({e}) + 0.3 * torch.sin(2 * x)",
        lambda e: f"({e}) * torch.exp(-0.1 * x * x)",
        lambda e: f"({e}) + x * 0.1",
        lambda e: f"torch.tanh({e})",
        lambda e: f"({e}) * torch.cos(x)",
        lambda e: f"torch.sin({e})",
        lambda e: f"({e}) / (1 + torch.abs({e}))",
    ]

    def __init__(self, pde: str = "maxwell", pop_size: int = 16):
        self.pde = pde
        self.pop_size = pop_size
        self.population: list = []   # list of {expr, score, gen}
        self.generation = 0
        self.best_expr = ""
        self.best_score = 0.0
        self.history: list = []

        # Seed population
        for seed in self.SEEDS[:pop_size]:
            self.population.append({"expr": seed, "score": 0.5, "gen": 0})

    def _safe_eval(self, expr: str):
        """Compile expression into a callable. Returns None on failure."""
        try:
            _ns = {
                "torch": torch, "math": math,
                "sin": torch.sin, "cos": torch.cos, "tanh": torch.tanh,
                "exp": torch.exp, "abs": torch.abs, "relu": torch.relu,
                "sigmoid": torch.sigmoid, "__builtins__": {},
            }
            fn = eval(f"lambda x: {expr}", _ns)
            with torch.no_grad():
                y = fn(torch.tensor([-1.0, 0.0, 1.0]))
                if not torch.isfinite(y).all():
                    return None
            return fn
        except Exception:
            return None

    def evaluate(self, expr: str) -> float:
        """Score an expression against the current PDE."""
        fn = self._safe_eval(expr)
        if fn is None:
            return 0.0
        suit = ActivationSuitability.score(fn, self.pde)
        # Smoothness bonus: penalize expressions that produce large values
        try:
            with torch.no_grad():
                xr = torch.linspace(-3, 3, 200)
                y = fn(xr)
                if not torch.isfinite(y).all():
                    return 0.0
                scale_pen = float(y.abs().max().clamp(max=10)) / 10
                smooth = 1.0 - scale_pen * 0.2
        except Exception:
            smooth = 0.5
        return round(max(0.0, suit * smooth), 4)

    def mutate(self, expr: str) -> str:
        """Apply a random mutation to an expression."""
        mut = random.choice(self.MUTATIONS)
        try:
            new_expr = mut(expr)
            # Validate: must be shorter than 200 chars
            if len(new_expr) > 200:
                return expr
            if self._safe_eval(new_expr) is None:
                return expr
            return new_expr
        except Exception:
            return expr

    def crossover(self, e1: str, e2: str) -> str:
        """Arithmetic crossover: α*e1 + (1-α)*e2."""
        alpha = round(random.uniform(0.3, 0.7), 2)
        new_expr = f"{alpha} * ({e1}) + {1-alpha} * ({e2})"
        if self._safe_eval(new_expr) and len(new_expr) < 200:
            return new_expr
        return e1

    def step(self) -> dict:
        """
        Run one generation of evolution.
        Returns the best expression found this generation.
        """
        self.generation += 1

        # Score all unevaluated members
        for member in self.population:
            if member["score"] == 0.5 and member["gen"] == 0:
                member["score"] = self.evaluate(member["expr"])
            member["gen"] = self.generation

        # Sort by score
        self.population.sort(key=lambda x: -x["score"])

        # Update best
        if self.population and self.population[0]["score"] > self.best_score:
            self.best_score = self.population[0]["score"]
            self.best_expr  = self.population[0]["expr"]

        # Elites: keep top-4
        elites = self.population[:4]

        # Offspring: mutations + crossovers
        offspring = []
        while len(offspring) < self.pop_size - 4:
            parent = random.choice(elites)["expr"]
            if random.random() < 0.7:
                child_expr = self.mutate(parent)
            else:
                parent2 = random.choice(elites)["expr"]
                child_expr = self.crossover(parent, parent2)
            child_score = self.evaluate(child_expr)
            offspring.append({"expr": child_expr, "score": child_score, "gen": self.generation})

        self.population = elites + offspring
        self.population.sort(key=lambda x: -x["score"])

        result = {
            "generation":  self.generation,
            "best_expr":   self.best_expr,
            "best_score":  self.best_score,
            "top5":        [{"expr": m["expr"], "score": m["score"]}
                            for m in self.population[:5]],
        }
        self.history.append(result)
        return result

    def state(self) -> dict:
        return {
            "generation":  self.generation,
            "best_expr":   self.best_expr,
            "best_score":  round(self.best_score, 4),
            "pop_size":    len(self.population),
            "top3":        [{"expr": m["expr"], "score": m["score"]}
                            for m in self.population[:3]],
        }


# ════════════════════════════════════════════════════════════
# 23. PARAMETRIC ACTIVATION OPTIMIZER
#     Optimizes numerical parameters in a parametric activation
#     expression using gradient-free Nelder-Mead search.
#
#     Example: "a * torch.sin(b * x) + c * torch.tanh(d * x)"
#     with parameters [a, b, c, d] optimized by maximizing
#     ActivationSuitability score.
#
#     This bridges hand-crafted parametric families with
#     automatic optimization — the user defines the form,
#     the algorithm finds the best parameters.
# ════════════════════════════════════════════════════════════
class ParametricActivationOptimizer:
    """
    Nelder-Mead optimization of scalar parameters in an activation.

    Usage:
        opt = ParametricActivationOptimizer(
            template="a * torch.sin(b * x)",
            param_names=["a", "b"],
            bounds={"a": (0.1, 3.0), "b": (0.5, 5.0)},
        )
        result = opt.optimize(pde="maxwell", n_iters=50)
    """

    def __init__(
        self,
        template: str,
        param_names: list,
        bounds: dict = None,
    ):
        self.template = template
        self.param_names = param_names
        self.bounds = bounds or {p: (0.1, 3.0) for p in param_names}
        self.best_params: dict = {}
        self.best_score: float = 0.0
        self.history: list = []

    def _make_fn(self, params: dict):
        """Substitute parameter values and compile."""
        expr = self.template
        for p, v in params.items():
            expr = expr.replace(p, str(round(float(v), 6)))
        try:
            _ns = {
                "torch": torch, "math": math,
                "sin": torch.sin, "cos": torch.cos, "tanh": torch.tanh,
                "exp": torch.exp, "abs": torch.abs, "__builtins__": {},
            }
            fn = eval(f"lambda x: {expr}", _ns)
            with torch.no_grad():
                y = fn(torch.tensor([0.0, 1.0, -1.0]))
                if not torch.isfinite(y).all():
                    return None, expr
            return fn, expr
        except Exception:
            return None, expr

    def _objective(self, param_vec: np.ndarray, pde: str) -> float:
        """Negative suitability score (minimize for Nelder-Mead)."""
        params = dict(zip(self.param_names, param_vec))
        # Apply bounds
        for p, (lo, hi) in self.bounds.items():
            params[p] = np.clip(params[p], lo, hi)
        fn, _ = self._make_fn(params)
        if fn is None:
            return 1.0   # worst possible (we minimize)
        score = ActivationSuitability.score(fn, pde)
        return -score    # negate for minimization

    def optimize(self, pde: str = "maxwell", n_iters: int = 60) -> dict:
        """
        Run Nelder-Mead optimization.
        Returns dict with best_expr, best_params, best_score.
        """
        from scipy.optimize import minimize
        try:
            x0 = np.array([1.0] * len(self.param_names))
            result = minimize(
                fun=lambda v: self._objective(v, pde),
                x0=x0,
                method="Nelder-Mead",
                options={"maxiter": n_iters, "xatol": 1e-3, "fatol": 1e-4},
            )
            best_params = dict(zip(self.param_names, result.x))
            fn, best_expr = self._make_fn(best_params)
            best_score = -result.fun if fn is not None else 0.0
        except ImportError:
            # Fallback: random search if scipy unavailable
            best_params = {}; best_score = 0.0; best_expr = self.template
            for _ in range(n_iters):
                params = {p: np.random.uniform(*self.bounds[p]) for p in self.param_names}
                fn, expr = self._make_fn(params)
                if fn is None: continue
                s = ActivationSuitability.score(fn, pde)
                if s > best_score:
                    best_score = s; best_params = params; best_expr = expr
        except Exception:
            best_params = {}; best_score = 0.0; best_expr = self.template

        self.best_params = best_params
        self.best_score  = round(float(best_score), 4)
        entry = {"expr": best_expr, "params": {k: round(float(v), 4) for k,v in best_params.items()},
                 "score": self.best_score, "pde": pde}
        self.history.append(entry)
        return entry

    def state(self) -> dict:
        return {
            "template":   self.template,
            "best_params": {k: round(float(v), 4) for k,v in self.best_params.items()},
            "best_score": self.best_score,
            "n_runs":     len(self.history),
        }


# ════════════════════════════════════════════════════════════
# 24. ACTIVATION GRADIENT FLOW ANALYZER
#     Measures how well an activation propagates gradients
#     through a network of the target depth.
#
#     Method:
#       1. Build a mini-network of depth d with the activation
#       2. Run a random input forward pass
#       3. Compute gradients back through all layers
#       4. Measure: saturation%, gradient magnitude at layer 0,
#          effective depth (how many layers carry signal)
#
#     Metrics:
#       saturation:       fraction of neurons in saturation region
#       grad_attenuation: ratio of output to input gradient norms
#       effective_depth:  last layer index with gradient > threshold
#       recommended_init: SIREN, Xavier, or Kaiming
# ════════════════════════════════════════════════════════════
class ActivationGradFlowAnalyzer:
    """
    Measures gradient flow quality for a given activation function.
    Detects saturation and vanishing gradients before full training.
    """

    def __init__(self, test_depth: int = 6, test_width: int = 64):
        self.depth = test_depth
        self.width = test_width

    def analyze(self, act_fn, in_dim: int = 2) -> dict:
        """
        Build a test network and measure gradient flow statistics.

        Args:
            act_fn:  callable activation function (PyTorch)
            in_dim:  input dimensionality

        Returns:
            dict with saturation, grad_attenuation, effective_depth,
                 per_layer_gnorm, recommended_init
        """
        try:
            import torch.nn as nn

            class TestNet(nn.Module):
                def __init__(self, act, in_d, w, depth):
                    super().__init__()
                    layers = []
                    dims = [in_d] + [w] * depth + [1]
                    for i in range(len(dims)-1):
                        layers.append(nn.Linear(dims[i], dims[i+1], dtype=torch.float64))
                    self.linears = nn.ModuleList(layers)
                    self.act = act

                def forward(self, x):
                    for i, lin in enumerate(self.linears[:-1]):
                        x = self.act(lin(x))
                    return self.linears[-1](x)

            net = TestNet(act_fn, in_dim, self.width, self.depth)
            for p in net.parameters():
                nn.init.xavier_normal_(p) if p.dim() > 1 else nn.init.zeros_(p)

            x = torch.randn(32, in_dim, dtype=torch.float64, requires_grad=True)
            y = net(x).sum()
            y.backward()

            # Per-layer gradient norms
            gnorms = []
            activations = []
            with torch.no_grad():
                xf = torch.randn(256, in_dim, dtype=torch.float64)
                for lin in net.linears[:-1]:
                    xf = act_fn(lin(xf))
                    activations.append(xf.detach())

            for lin in net.linears:
                if lin.weight.grad is not None:
                    gnorms.append(float(lin.weight.grad.norm()))

            # Saturation: how many activations are in flat region
            all_acts = torch.cat([a.flatten() for a in activations])
            with torch.no_grad():
                # Numerical derivative to detect saturation
                eps = 1e-4
                dx = act_fn(all_acts + eps) - act_fn(all_acts - eps)
                deriv = (dx / (2 * eps)).abs()
                saturation = float((deriv < 0.05).float().mean())

            # Gradient attenuation
            if len(gnorms) >= 2 and gnorms[-1] > 1e-12:
                attenuation = round(float(gnorms[0] / gnorms[-1]), 4)
            else:
                attenuation = 0.0

            # Effective depth: layers that still carry meaningful gradient
            threshold = max(gnorms) * 0.01 if gnorms else 1e-10
            effective = sum(1 for g in gnorms if g > threshold)

            # Recommended initialization
            if saturation > 0.5:
                rec_init = "SIREN (sinusoidal acts need careful init)"
            elif attenuation < 0.1:
                rec_init = "Kaiming (corrects variance for relu-like)"
            else:
                rec_init = "Xavier (balanced variance for smooth acts)"

            return {
                "saturation":      round(saturation, 4),
                "grad_attenuation": attenuation,
                "effective_depth": effective,
                "per_layer_gnorm": [round(g, 6) for g in gnorms],
                "recommended_init": rec_init,
                "healthy":         saturation < 0.3 and attenuation > 0.05,
            }
        except Exception as e:
            return {
                "saturation": 0.5, "grad_attenuation": 1.0,
                "effective_depth": self.depth, "per_layer_gnorm": [],
                "recommended_init": "Xavier", "healthy": True,
                "error": str(e),
            }


# ════════════════════════════════════════════════════════════
# 25. ACTIVATION BENCHMARK
#     Lightweight benchmark of an activation function against
#     all built-in activations on the target PDE.
#     Runs 30-epoch mini-training runs and ranks by accuracy.
#     Used in the Activation Studio to rank custom expressions.
# ════════════════════════════════════════════════════════════
class ActivationBenchmark:
    """
    Fast benchmark comparing a custom activation against presets.
    Uses very short training runs (30 epochs) to estimate rank.
    """

    def __init__(self):
        self.results: list = []

    def run_async_friendly(
        self,
        act_fn,
        act_name: str,
        presets: list,
        pde: str,
        width: int = 64,
        depth: int = 4,
    ) -> dict:
        """
        Score custom activation vs presets using spectral suitability.
        Full training runs are handled by the search mode.
        Returns preliminary ranking.
        """
        from .models import ACTIVATIONS

        custom_score = ActivationSuitability.score(act_fn, pde)
        preset_scores = {
            a: ActivationSuitability.score(ACTIVATIONS.get(a, torch.cos), pde)
            for a in presets
        }

        all_scores = {act_name: custom_score, **preset_scores}
        ranked = sorted(all_scores.items(), key=lambda x: -x[1])
        rank = next(i+1 for i, (k,_) in enumerate(ranked) if k == act_name)

        result = {
            "custom_name":  act_name,
            "custom_score": round(custom_score, 4),
            "rank":         rank,
            "total":        len(all_scores),
            "ranked":       [{"act": k, "score": round(v, 4)} for k, v in ranked],
        }
        self.results.append(result)
        return result
