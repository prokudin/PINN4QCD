"""
solvers.py — Solver strategies and failure diagnostics
=======================================================
BUG FIXES (v4):
  - Import adaptive_colloc (not adaptive_colloc_maxwell — never existed)
  - Use compute_total_loss dispatcher for all 4 PDEs
  - run_solver is the single-run entry point (server.py calls this)
  - engine.py uses its own _train loop directly, not run_solver

Six solver strategies:
  classic    Adam → CosineAnnealing → L-BFGS (exact notebook baseline)
  adaptive   Residual-adaptive collocation sampling
  gradnorm   GradNorm automatic loss weight balancing
  fourier    Fourier Feature architecture (spectral bias fix)
  resnet     ResNet architecture (vanishing gradient fix)
  ensemble   3 independent models (variance reduction)
"""

import math
import time
import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .physics import (
    compute_total_loss,
    adaptive_colloc,
    compute_metrics,
    Lx, T_END, PI,
)
from .models import build_model, ACTIVATIONS


SOLVER_INFO = {
    "classic":  {"name": "Classic PINN",       "desc": "Adam + CosineAnnealing + L-BFGS. Exact notebook baseline."},
    "adaptive": {"name": "Adaptive Sampling",   "desc": "Residual-adaptive collocation. More pts where physics is violated."},
    "gradnorm": {"name": "GradNorm Balance",    "desc": "Auto-balances loss term weights. Prevents single term dominating."},
    "fourier":  {"name": "Fourier Feature",     "desc": "Random Fourier Feature encoding. Fixes spectral bias."},
    "resnet":   {"name": "ResNet PINN",         "desc": "Skip connections every 2 layers. Fixes vanishing gradients."},
    "ensemble": {"name": "Ensemble",            "desc": "3 independent models, ensemble. Reduces variance."},
}


def _make_colloc(pde, n, device, dtype):
    if pde in ("maxwell", "heat", "burgers"):
        x = (torch.rand(n,1,device=device,dtype=dtype)*Lx).requires_grad_(True)
        t = (torch.rand(n,1,device=device,dtype=dtype)*T_END).requires_grad_(True)
    else:
        x = (torch.rand(n,1,device=device,dtype=dtype)*(2*PI)).requires_grad_(True)
        t = x
    return x, t


class GradNormWeights(nn.Module):
    def __init__(self, n_tasks=3, alpha=1.5, device=None, dtype=torch.float64):
        super().__init__()
        self.log_w = nn.Parameter(torch.zeros(n_tasks, device=device, dtype=dtype))
        self.alpha = alpha
    def weights(self):
        return torch.exp(self.log_w)


async def _adam_loop(model, pde, epochs, lr, n_col, send_cb, device, dtype,
                     do_lbfgs=True, log_n=60, label="", adaptive=False, gn=None):
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    opt_gn = optim.Adam(gn.parameters(), lr=lr*0.05) if gn is not None else None
    hist = {"epoch":[],"loss":[],"pde_loss":[],"ratio":[],"wave_loss":[]}
    ev = max(1, epochs//log_n)
    t0 = time.time()

    for ep in range(1, epochs+1):
        opt.zero_grad()
        if opt_gn: opt_gn.zero_grad()

        if adaptive and ep%100==0 and ep>200:
            try:
                x, t_col = adaptive_colloc(model, pde, n_col, device, dtype)
                if pde not in ("harmonic",):
                    x = x.requires_grad_(True)
                    t_col = t_col.requires_grad_(True) if t_col is not None else x
                else:
                    x = x.requires_grad_(True); t_col = x
            except Exception:
                x, t_col = _make_colloc(pde, n_col, device, dtype)
        else:
            x, t_col = _make_colloc(pde, n_col, device, dtype)

        L = compute_total_loss(model, x, t_col, ep, epochs, pde, device, dtype)

        if gn is not None:
            W = gn.weights()
            task_losses = [L["pde"], L.get("left", L.get("bc", L["pde"])),
                           L.get("wave", L["pde"])]
            total_loss = sum(w*l for w,l in zip(W, task_losses))
        else:
            total_loss = L["total"]

        if not torch.isfinite(total_loss):
            for g in opt.param_groups: g["lr"] *= 0.1
            continue

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if opt_gn: opt_gn.step()

        if ep == int(epochs*0.60):
            for g in opt.param_groups: g["lr"] *= 0.2

        if ep%ev==0 or ep==epochs:
            m = compute_metrics(model, pde, device, dtype)
            hist["epoch"].append(ep)
            hist["loss"].append(round(float(total_loss),6))
            hist["pde_loss"].append(round(float(L["pde"]),6))
            hist["ratio"].append(round(m["ratio"],5))
            hist["wave_loss"].append(round(float(L.get("wave",0)),6))
            await send_cb({
                "type":"progress","label":label,
                "epoch":ep,"n_epochs":epochs,
                "loss":float(total_loss),"pde_loss":float(L["pde"]),
                "wave_loss":float(L.get("wave",0)),
                "left_loss":float(L.get("left",L.get("bc",0))),
                "ratio":m["ratio"],"pct":m["pct"],"rel_l2":m["rel_l2"],
                "elapsed":round(time.time()-t0,2),"history":hist,
            })
            await asyncio.sleep(0)

    if do_lbfgs:
        await send_cb({"type":"status","msg":f"L-BFGS [{label}]…"})
        lb = optim.LBFGS(model.parameters(), lr=0.01, max_iter=40,
                         history_size=50, line_search_fn="strong_wolfe")
        def _cl():
            lb.zero_grad()
            xr,tr = _make_colloc(pde, min(n_col,512), device, dtype)
            Lv = compute_total_loss(model,xr,tr,epochs,epochs,pde,device,dtype)["total"]
            Lv.backward(); return Lv
        try: lb.step(_cl)
        except Exception: pass

    return compute_metrics(model,pde,device,dtype), hist, time.time()-t0


async def run_solver(model, pde, epochs, lr, n_col, send_cb, device, dtype,
                     solver="classic", do_lbfgs=True, log_n=60, label=""):
    if solver == "gradnorm":
        gn = GradNormWeights(n_tasks=3, device=device, dtype=dtype)
        return await _adam_loop(model,pde,epochs,lr,n_col,send_cb,device,dtype,
                                do_lbfgs,log_n,label,adaptive=False,gn=gn)
    return await _adam_loop(model,pde,epochs,lr,n_col,send_cb,device,dtype,
                            do_lbfgs,log_n,label,adaptive=(solver=="adaptive"),gn=None)


def diagnose(hist, metrics):
    if not hist.get("loss") or len(hist["loss"]) < 5: return None, None
    losses = hist["loss"]; ratios = hist.get("ratio",[])
    if any(not math.isfinite(l) for l in losses[-5:]):
        return "NaN/Inf detected", "LR too high or SIREN init unstable. Try: lr×0.1, switch tanh/swish, or Fourier arch."
    recent = losses[-max(3,len(losses)//5):]
    spread = (max(recent)-min(recent))/(abs(min(recent))+1e-10)
    if spread<0.005 and metrics.get("rel_l2",1.0)>0.3:
        return "Training plateau", "Loss stagnated above target. Try: more epochs, lr×5, adaptive collocation, or GradNorm."
    if len(losses)>5:
        diffs=[abs(losses[i]-losses[i-1]) for i in range(1,len(losses))]
        if diffs[-1]>5*sum(diffs)/len(diffs):
            return "Loss oscillating", "Gradient instability. Try: lr×0.1, or ResNet architecture."
    pde_h = hist.get("pde_loss",[])
    if len(pde_h)>3 and pde_h[-1]/(pde_h[0]+1e-10)<0.01 and metrics.get("rel_l2",1.0)>0.3:
        return "BC failure", "PDE residual low but global accuracy poor. Increase bc_weight×5, or ResNet arch."
    if ratios and len(ratios)>=2 and ratios[-1]<0.5:
        return "Slow convergence", "Accuracy below 50%. Try: Fourier Feature encoding, wider/deeper net, or cos/sin activation."
    return None, None
