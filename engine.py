"""
engine.py v4 — Self-Directing Autonomous Research Engine
=========================================================
20 algorithms in a closed feedback loop.

Auto-correction decisions:
  NTK κ > 1e6       → override arch to ResNet, depth ≤ 5
  high_freq=True    → override arch to Fourier, σ from FourierSigmaTuner
  stiff detected    → increase collocation pts × 2
  PopulationEntropy < 0.2 → force random injection
  PersistentConflict → apply PCGrad every 10 epochs (not 50)
  ActivationSuitability low → prefer higher-scoring activations next gen
  NaN × 3           → abort run, record failure, skip
  Plateau           → CurriculumScheduler triggers LR drop
"""

import math, time, copy, random, asyncio, json
from collections import defaultdict

import torch
import torch.nn as nn

from .physics   import (compute_metrics, vis_2d, field_grid, compare_curve,
                        compute_total_loss, adaptive_colloc, get_residual_field,
                        PDE_REGISTRY, Lx, T_END, PI)
from .models    import build_model, ACTIVATIONS
from .solvers   import diagnose, SOLVER_INFO
from .algorithms import (
    BayesianOptimizer, ConfigScorer, MultiArmedBandit,
    PCGrad, GradientOrthogonality, HomoscedasticWeighter, AdaptiveLossWeighter,
    CurriculumScheduler, ReplayBuffer, ResidualGuidedSampler,
    SpectralAnalyzer, FourierSigmaTuner, NTKMonitor, ActivationSuitability,
    ParetoTracker, NoveltySearch, FailureMemory, PopulationEntropy,
    MetaLearner, SelfDistiller, LayerHealthMonitor,
)

_ACTS   = [a for a in ACTIVATIONS.keys() if a != "custom"]
_WIDTHS = [32,64,96,128,192,256,320]
_DEPTHS = [3,4,5,6,7,8]
_LRS    = [0.04,0.02,0.01,5e-3,1e-3,5e-4,1e-4]
_ARCHS  = ["standard","fourier","resnet"]
_SOLS   = ["classic","adaptive","gradnorm"]


# ════════════════════════════════════════════════════════════
# CORE TRAINING LOOP
# Integrates all algorithm hooks at the correct points.
# ════════════════════════════════════════════════════════════

async def _train(model, pde, epochs, lr, n_col, send_cb, device, dtype,
                 solver="classic", do_lbfgs=True, label="",
                 curriculum=None, distiller=None, replay=None,
                 homo_w=None, adaptive_w=None, meta=None,
                 grad_ortho=None, rg_sampler=None, phase=1):

    if curriculum is None: curriculum = CurriculumScheduler()
    layer_mon  = LayerHealthMonitor()
    pcgrad_interval = 10 if (grad_ortho and grad_ortho.is_persistent()) else 50

    # Meta warm-start for Phase 2+
    if meta is not None and phase >= 2:
        meta.warm_start(model, noise=0.015)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    opt_hw = torch.optim.Adam(homo_w.parameters(), lr=lr*0.05) if homo_w else None

    hist = {"epoch":[],"loss":[],"pde_loss":[],"ratio":[],"wave_loss":[]}
    ev   = max(1, epochs//60)
    t0   = time.time()
    nan_restarts = 0
    layer_flags_latest = []

    def _colloc():
        if pde in ("maxwell","heat","burgers"):
            x=(torch.rand(n_col,1,device=device,dtype=dtype)*Lx).requires_grad_(True)
            t=(torch.rand(n_col,1,device=device,dtype=dtype)*T_END).requires_grad_(True)
        else:
            x=(torch.rand(n_col,1,device=device,dtype=dtype)*(2*PI)).requires_grad_(True)
            t=x
        return x,t

    for ep in range(1, epochs+1):
        opt.zero_grad()
        if opt_hw: opt_hw.zero_grad()

        # ── Adaptive collocation: residual-guided sampler (Phase 2+)
        #    or standard adaptive_colloc (Phase 1 adaptive solver)
        if rg_sampler is not None and ep>100 and ep%50==0:
            try: rg_sampler.build(model, pde, device, dtype)
            except: pass

        if rg_sampler is not None and rg_sampler.density is not None and ep>100:
            xs,ts=rg_sampler.sample(n_col, device, dtype)
            if xs is None: xs,ts=_colloc()
        elif solver=="adaptive" and ep%100==0 and ep>200:
            try:
                xs,ts=adaptive_colloc(model,pde,n_col,device,dtype)
                xs=xs.requires_grad_(True)
                if pde not in ("harmonic",): ts=ts.requires_grad_(True) if ts is not None else xs
            except: xs,ts=_colloc()
        else:
            xs,ts=_colloc()

        # ── Replay buffer injection
        if replay is not None and ep>200:
            rb=replay.sample(n_col//4, device, dtype)
            if rb is not None:
                if rb.shape[-1]==2:
                    rx,rt=rb[:,0:1].requires_grad_(True),rb[:,1:2].requires_grad_(True)
                    xs=torch.cat([xs,rx],0); ts=torch.cat([ts,rt],0)
                else:
                    xb=rb.requires_grad_(True)
                    xs=torch.cat([xs,xb],0); ts=xs

        # ── Forward + loss
        w_bc=curriculum.bc_w()
        L=compute_total_loss(model,xs,ts,ep,epochs,pde,device,dtype,w_bc)

        # ── NaN guard
        if not torch.isfinite(L["total"]):
            nan_restarts+=1
            if nan_restarts>3: break
            for g in opt.param_groups: g["lr"]*=0.1
            continue

        # ── Loss weighting: Homoscedastic (replaces manual w_bc after warmup)
        if homo_w is not None and ep>100:
            task_l=[L["pde"],L.get("left",L.get("bc",L["pde"])),L.get("wave",L["pde"])]
            total_loss=homo_w(task_l)
            if opt_hw: opt_hw.zero_grad()
        elif adaptive_w is not None and ep>50:
            task_l=[L["pde"],L.get("left",L.get("bc",L["pde"])),L.get("wave",L["pde"])]
            adaptive_w.update(model,task_l,opt)
            total_loss=adaptive_w.apply(task_l)
        else:
            total_loss=L["total"]

        # ── PCGrad (Phase 2+, every N epochs based on conflict history)
        if phase>=2 and ep%pcgrad_interval==0:
            key_l=[v for k,v in L.items() if k!="total" and
                   isinstance(v,torch.Tensor) and v.requires_grad]
            if len(key_l)>=2:
                try:
                    angles=PCGrad.conflict_angles(model, key_l[:3])
                    if grad_ortho: grad_ortho.record(angles)
                    PCGrad.apply(model, key_l[:3])
                except: total_loss.backward()
            else: total_loss.backward()
        else:
            total_loss.backward()

        # ── Distillation supplement
        if distiller is not None and ep>50:
            Xb=torch.cat([xs,ts],1) if pde!="harmonic" else xs
            try: distiller.dist_loss(model,Xb.detach()).backward()
            except: pass

        # ── Layer health + gradient clip
        lh=layer_mon.check(model)
        layer_flags_latest=lh.get("flags",[])
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if opt_hw: opt_hw.step()

        # ── Hard LR drop at 60%
        if ep==int(epochs*0.60):
            for g in opt.param_groups: g["lr"]*=0.2

        # ── Curriculum update
        adj=curriculum.update(L)
        if adj.get("reduce_lr"):
            for g in opt.param_groups: g["lr"]*=0.5

        # ── Push high-residual points to replay buffer
        if replay is not None and ep%50==0:
            try:
                res,xf=get_residual_field(model,pde,device,dtype,n=256)
                pts=(torch.cat([xf.view(-1,1),torch.zeros_like(xf.view(-1,1))],1)
                     if pde!="harmonic" else xf.view(-1,1))
                replay.push(pts,res)
            except: pass

        # ── Progress emit
        if ep%ev==0 or ep==epochs:
            m=compute_metrics(model,pde,device,dtype)
            hist["epoch"].append(ep); hist["loss"].append(round(float(total_loss),6))
            hist["pde_loss"].append(round(float(L["pde"]),6))
            hist["ratio"].append(round(m["ratio"],5))
            hist["wave_loss"].append(round(float(L.get("wave",0)),6))
            await send_cb({
                "type":"progress","label":label,"phase":phase,
                "epoch":ep,"n_epochs":epochs,
                "loss":float(total_loss),"pde_loss":float(L["pde"]),
                "wave_loss":float(L.get("wave",0)),
                "left_loss":float(L.get("left",L.get("bc",0))),
                "ratio":m["ratio"],"pct":m["pct"],"rel_l2":m["rel_l2"],
                "elapsed":round(time.time()-t0,2),"history":hist,
                "curriculum":curriculum.state(),
                "layer_flags":layer_flags_latest,
                "homo_weights":homo_w.weights() if homo_w else None,
                "adaptive_weights":adaptive_w.state() if adaptive_w else None,
                "pcgrad_interval":pcgrad_interval,
            })
            await asyncio.sleep(0)

    # ── L-BFGS refinement
    if do_lbfgs:
        await send_cb({"type":"status","msg":f"L-BFGS [{label}]…"})
        lb=torch.optim.LBFGS(model.parameters(),lr=0.01,max_iter=40,
                              history_size=50,line_search_fn="strong_wolfe")
        def _cl():
            lb.zero_grad()
            xr,tr=_colloc()
            Lv=compute_total_loss(model,xr,tr,epochs,epochs,pde,device,dtype)["total"]
            Lv.backward(); return Lv
        try: lb.step(_cl)
        except: pass

    return compute_metrics(model,pde,device,dtype), hist, time.time()-t0


# ════════════════════════════════════════════════════════════
# POPULATION GENERATOR
# HOF mutations + UCB1 bandit + Bayesian suggestion
# + ActivationSuitability bias + PopulationEntropy diversity injection
# ════════════════════════════════════════════════════════════

def _random_cfg(pde):
    return {"act":random.choice(_ACTS),"arch":random.choice(_ARCHS),
            "width":random.choice(_WIDTHS),"depth":random.choice(_DEPTHS),
            "lr":random.choice(_LRS),"solver":random.choice(_SOLS),"pde":pde}

def _mutate(cfg, pde):
    c=dict(cfg); c["pde"]=pde
    k=random.choice(["act","arch","width","depth","lr","solver"])
    if   k=="act":    c["act"]   =random.choice(_ACTS)
    elif k=="arch":   c["arch"]  =random.choice(_ARCHS)
    elif k=="width":  c["width"] =random.choice(_WIDTHS)
    elif k=="depth":  c["depth"] =random.choice(_DEPTHS)
    elif k=="lr":     c["lr"]    =random.choice(_LRS)
    elif k=="solver": c["solver"]=random.choice(_SOLS)
    return c

def _generate_pop(n, pde, hof, bayes, failure, scorer, bandit,
                  pop_entropy, act_suit, sigma_tuner, ntk_kappa):
    pool=[]

    # 1. HOF mutations (top 3 configs, each mutated once)
    for h in hof[:3]:
        c=_mutate(h.get("cfg",{}), pde)
        if not failure.blacklisted(c): pool.append(c)

    # 2. Bayesian suggestion
    rand_cands=[_random_cfg(pde) for _ in range(40)]
    pool.append(bayes.suggest(rand_cands))

    # 3. Activation-suitability biased selection
    #    Favor activations with high spectral overlap for this PDE
    for act in random.sample(_ACTS, min(4, len(_ACTS))):
        try:
            fn=ACTIVATIONS.get(act)
            if fn is None: continue
            score=ActivationSuitability.score(fn, pde)
            if score>0.3:
                c=_random_cfg(pde); c["act"]=act
                if not failure.blacklisted(c): pool.append(c)
        except: pass

    # 4. UCB1 bandit — favor high-performing (act, arch) arms
    for _ in range(n*2):
        act=random.choice(_ACTS); arch=random.choice(_ARCHS)
        if bandit.ucb1(act,arch)>0:
            c=_random_cfg(pde); c["act"]=act; c["arch"]=arch
            if not failure.blacklisted(c): pool.append(c)

    # 5. NTK override: if recent κ severe, inject ResNet configs
    if ntk_kappa and ntk_kappa>1e6:
        for _ in range(3):
            c=_random_cfg(pde); c["arch"]="resnet"; c["depth"]=min(c["depth"],5)
            pool.append(c)

    # 6. Fourier override: if spectral analyzer recommends high-σ
    if sigma_tuner.get()>3.0:
        for _ in range(2):
            c=_random_cfg(pde); c["arch"]="fourier"
            pool.append(c)

    # 7. Population entropy: if collapsed, inject random diversity
    if pop_entropy.is_collapsed():
        for _ in range(max(n//3,2)):
            pool.append(_random_cfg(pde))

    # 8. Pad to n*3, then ConfigScorer filter
    while len(pool)<n*3:
        c=_random_cfg(pde)
        if not failure.blacklisted(c): pool.append(c)
    pool=scorer.filter_top(pool, keep=0.6)
    random.shuffle(pool)
    return pool[:n]


# ════════════════════════════════════════════════════════════
# POST-RUN ANALYSIS
# Runs all analysis algorithms on a completed training.
# Returns dict of all analysis results.
# ════════════════════════════════════════════════════════════

def _analyze(model, pde, device, dtype, label, cfg, metrics, elapsed,
             run_id, bayes, failure, pareto, novelty, sigma_tuner,
             scorer, meta, bandit, pop_entropy, act_suit):
    spec={}; ntk_k=-1.0; novelty_sc=0.0; suit_sc=0.5

    # Spectral analysis
    try:
        res,xf=get_residual_field(model,pde,device,dtype)
        spec=SpectralAnalyzer.analyze(res,xf)
        sigma_tuner.update(spec)
    except: pass

    # NTK condition number
    try:
        in_d=2 if pde=="maxwell" else 1
        Xb=torch.rand(32,in_d,device=device,dtype=dtype)
        ntk_k=NTKMonitor.condition_number(model,Xb)
    except: pass

    # Novelty score
    cmp={}
    try:
        cmp=compare_curve(model,pde,device,dtype)
        curve=cmp.get("pred",cmp.get("E_pred",[]))
        novelty_sc=novelty.novelty(curve)
        novelty.add(curve,{"label":label})
    except: pass

    # Activation suitability
    try:
        fn=ACTIVATIONS.get(cfg.get("act","cos"))
        if fn: suit_sc=ActivationSuitability.score(fn, pde)
    except: pass

    score=1.0-metrics.get("rel_l2",1.0)
    bayes.observe(cfg, score)
    scorer.observe(cfg, score)
    bandit.update(cfg.get("act","cos"), cfg.get("arch","standard"), score)

    if metrics["rel_l2"]>0.85: failure.record_fail(cfg)
    else:
        failure.record_win(cfg, score)
        meta.record(model, score)

    try: pareto.add(cfg, metrics, elapsed, model.n_params(), run_id, label)
    except: pass

    ntk_status, ntk_advice = NTKMonitor.diagnose(ntk_k)

    return {
        "spec":       spec,
        "ntk_kappa":  ntk_k,
        "ntk_status": ntk_status,
        "ntk_advice": ntk_advice,
        "novelty":    novelty_sc,
        "suitability":suit_sc,
        "compare":    cmp,
    }


# ════════════════════════════════════════════════════════════
# HOF SNAPSHOT (for leaderboard streaming)
# ════════════════════════════════════════════════════════════

def _hof_snap(hof, n=15):
    return [{
        "rank":i+1,"label":h["cfg"].get("act","?"),
        "arch":h["cfg"].get("arch","std"),"solver":h["cfg"].get("solver","cls"),
        "width":h["cfg"].get("width",0),"depth":h["cfg"].get("depth",0),
        "lr":h["cfg"].get("lr",0),"pde":h.get("pde","?"),
        "ratio":round(h["metrics"]["ratio"],4),"rel_l2":round(h["metrics"]["rel_l2"],5),
        "elapsed":round(h.get("elapsed",0),1),"gen":h.get("gen",0),
        "phase":h.get("phase","scan"),"novelty":round(h.get("novelty",0),3),
        "ntk_kappa":h.get("ntk_kappa",-1),"ntk_status":h.get("ntk_status","—"),
        "suitability":round(h.get("suitability",0.5),3),
        "spectral":h.get("spec",{}),"note":h.get("note",""),
    } for i,h in enumerate(hof[:n])]


# ════════════════════════════════════════════════════════════
# AUTONOMOUS ENGINE  —  main WebSocket-streaming loop
# ════════════════════════════════════════════════════════════

async def stream_autonomous(ws, cfg_in: dict, device, dtype):
    pde         = cfg_in.get("pde",           "maxwell")
    p1_ep       = int(cfg_in.get("phase1_epochs", 300))
    p2_ep       = int(cfg_in.get("phase2_epochs",1500))
    champ_ep    = int(cfg_in.get("champ_epochs",  3000))
    pop_size    = int(cfg_in.get("pop_size",        6))
    max_gen     = int(cfg_in.get("max_gen",      9999))
    n_col       = int(cfg_in.get("n_colloc",     1024))

    # ── Instantiate all 20 algorithms ───────────────────────
    bayes       = BayesianOptimizer()
    scorer      = ConfigScorer()
    bandit      = MultiArmedBandit()
    grad_ortho  = GradientOrthogonality()
    sigma_tuner = FourierSigmaTuner()
    pareto      = ParetoTracker()
    novelty     = NoveltySearch()
    failure     = FailureMemory()
    pop_ent     = PopulationEntropy()
    meta        = MetaLearner()
    replay      = ReplayBuffer(capacity=3000)
    rg_sampler  = ResidualGuidedSampler()
    act_suit    = ActivationSuitability()

    hof=[]
    best_ever=None; teacher=None
    total_runs=[0]; start_t=time.time()
    ntk_kappa_latest=-1.0
    compare_all={"x":None,"true":None,"curves":{}}
    spectral_hist=[]; ntk_hist=[]

    def rt():
        s=int(time.time()-start_t); return f"{s//3600}h{(s%3600)//60}m{s%60}s"

    async def send(d):
        try: await ws.send_text(json.dumps(d))
        except: pass

    def update_hof(cfg_, metrics, elapsed, gen, run_id, phase, extras={}):
        nonlocal best_ever
        entry=dict(extras)
        entry.update({"cfg":cfg_,"metrics":metrics,"elapsed":elapsed,
                      "gen":gen,"run_id":run_id,"phase":phase})
        hof.append(entry)
        hof.sort(key=lambda x:x["metrics"]["rel_l2"])
        del hof[30:]
        if best_ever is None or metrics["rel_l2"]<best_ever["metrics"]["rel_l2"]:
            best_ever=entry

    # ── Announce start with full algorithm manifest ──────────
    await send({
        "type":"autonomous_start","pde":pde,"pop_size":pop_size,
        "algorithms":{
            "BayesianOptimizer":     {"desc":"GP surrogate + EI/UCB/Thompson","active":True},
            "ConfigScorer":          {"desc":"kNN pre-filter","active":True},
            "MultiArmedBandit":      {"desc":"UCB1 over act×arch arms","active":True},
            "PCGrad":                {"desc":"Gradient surgery","active":True},
            "GradientOrthogonality": {"desc":"Conflict angle tracking","active":True},
            "HomoscedasticWeighter": {"desc":"Uncertainty loss weights","active":True},
            "AdaptiveLossWeighter":  {"desc":"MOAT gradient-magnitude scaling","active":True},
            "CurriculumScheduler":   {"desc":"BC weight + LR + wave activation","active":True},
            "ReplayBuffer":          {"desc":"Prioritized residual replay","active":True},
            "ResidualGuidedSampler": {"desc":"2D importance field sampling","active":True},
            "SpectralAnalyzer":      {"desc":"FFT residual → Fourier σ","active":True},
            "FourierSigmaTuner":     {"desc":"EMA σ","active":True},
            "NTKMonitor":            {"desc":"NTK κ → arch override","active":True},
            "ActivationSuitability": {"desc":"Spectral overlap scoring","active":True},
            "ParetoTracker":         {"desc":"Multi-objective front","active":True},
            "NoveltySearch":         {"desc":"Behavioral diversity k-NN","active":True},
            "FailureMemory":         {"desc":"Gaussian blacklist","active":True},
            "PopulationEntropy":     {"desc":"Diversity entropy","active":True},
            "MetaLearner":           {"desc":"HOF weighted warm-start","active":True},
            "SelfDistiller":         {"desc":"Champion→student KD","active":True},
        }
    })

    for gen in range(max_gen):

        population=_generate_pop(pop_size,pde,hof,bayes,failure,scorer,bandit,
                                  pop_ent,act_suit,sigma_tuner,ntk_kappa_latest)
        pop_entropy_val=pop_ent.measure(population)

        await send({
            "type":"generation_start","gen":gen,"runtime":rt(),
            "n_hof":len(hof),"pop_entropy":pop_entropy_val,
            "bayes":bayes.state(),"failure":failure.state(),
            "meta":meta.state(),"replay":replay.state(),
            "sigma":sigma_tuner.state(),"bandit":bandit.state(),
            "pareto_front":pareto.front()[:5],
            "grad_ortho":grad_ortho.state(),
        })

        # ══════════════════════════════════════════════════════
        # PHASE 1 — Rapid scan
        # ══════════════════════════════════════════════════════
        await send({"type":"phase","phase":1,"gen":gen,
                    "label":f"Gen {gen} · Phase 1 · {len(population)} configs × {p1_ep} ep"})
        gen_results=[]

        for i,pcfg in enumerate(population):
            total_runs[0]+=1; rid=total_runs[0]
            lbl=f"g{gen}r{i}·{pcfg.get('act','?')[:5]}·{pcfg.get('arch','std')[:3]}"
            await send({"type":"run_start","gen":gen,"run":i,"total":len(population),
                        "config":pcfg,"label":lbl,"run_id":rid,"phase":1})

            model=build_model(pde,pcfg.get("act","cos"),pcfg.get("width",128),
                               pcfg.get("depth",5),pcfg.get("arch","standard"),device,dtype)
            curr=CurriculumScheduler()

            async def _cb(d,_l=lbl): 
                if d.get("type") in ("progress","status"): await send({**d,"label":_l})

            try:
                metrics,hist,elapsed=await _train(
                    model,pde,p1_ep,pcfg.get("lr",1e-3),n_col,_cb,device,dtype,
                    solver=pcfg.get("solver","classic"),do_lbfgs=False,label=lbl,
                    curriculum=curr,replay=replay,phase=1)
            except Exception as e:
                await send({"type":"run_error","gen":gen,"run":i,"label":lbl,"msg":str(e)})
                failure.record_fail(pcfg); continue

            anl=_analyze(model,pde,device,dtype,lbl,pcfg,metrics,elapsed,rid,
                         bayes,failure,pareto,novelty,sigma_tuner,scorer,meta,
                         bandit,pop_ent,act_suit)
            ntk_kappa_latest=anl["ntk_kappa"]
            spectral_hist.append(anl["spec"]); ntk_hist.append(anl["ntk_kappa"])

            # Update compare overlay
            if anl["compare"].get("x"):
                compare_all["x"]    = compare_all["x"]    or anl["compare"]["x"]
                compare_all["true"] = compare_all["true"] or anl["compare"].get("true",anl["compare"].get("E_true"))
            curve=anl["compare"].get("pred") or anl["compare"].get("E_pred")
            if curve: compare_all["curves"][lbl]=curve

            # NTK / spectral auto-correction note
            note=""
            if anl["ntk_kappa"]>1e6: note="→ResNet(NTK severe)"
            elif anl["spec"].get("high_freq"): note=f"→Fourier(σ={sigma_tuner.get():.1f})"

            diag_t,diag_m=diagnose(hist,metrics)
            update_hof(pcfg,metrics,elapsed,gen,rid,"scan",{
                "spec":anl["spec"],"ntk_kappa":anl["ntk_kappa"],
                "ntk_status":anl["ntk_status"],"novelty":anl["novelty"],
                "suitability":anl["suitability"],"note":note})

            gen_results.append({"cfg":pcfg,"metrics":metrics,"hist":hist,
                                 "elapsed":elapsed,"label":lbl,"anl":anl,"note":note})
            gen_results.sort(key=lambda r:r["metrics"]["rel_l2"])

            await send({
                "type":"run_done","gen":gen,"run":i,"label":lbl,"phase":1,
                "metrics":metrics,"elapsed":elapsed,"run_id":rid,
                "hall_of_fame":_hof_snap(hof),
                "pareto_front":pareto.front()[:5],
                "diag_title":diag_t,"diag_msg":diag_m,
                "compare_data":compare_all,
                "spectral":anl["spec"],"ntk_kappa":anl["ntk_kappa"],
                "ntk_status":anl["ntk_status"],"ntk_advice":anl["ntk_advice"],
                "novelty":anl["novelty"],"suitability":anl["suitability"],
                "arch_note":note,"curriculum":curr.state(),
                "bayes":bayes.state(),"bandit":bandit.state(),
                "failure":failure.state(),"replay":replay.state(),
                "sigma":sigma_tuner.state(),"meta":meta.state(),
                "total_runs":total_runs[0],"runtime":rt(),
                "pop_entropy":pop_entropy_val,
            })
            await asyncio.sleep(0)

        # ══════════════════════════════════════════════════════
        # PHASE 2 — Deep retrain: top-2 + highest novelty
        # ══════════════════════════════════════════════════════
        top_n=gen_results[:min(2,len(gen_results))]
        for r in gen_results[2:5]:
            if r["anl"]["novelty"]>0.55: top_n.append(r); break

        await send({"type":"phase","phase":2,"gen":gen,
                    "label":f"Gen {gen} · Phase 2 · {len(top_n)} configs × {p2_ep} ep + all algorithms"})

        for r in top_n:
            pcfg=r["cfg"]
            # Apply NTK/spectral overrides for Phase 2
            if r["anl"]["ntk_kappa"]>1e6:
                pcfg=dict(pcfg); pcfg["arch"]="resnet"; pcfg["depth"]=min(pcfg.get("depth",5),5)
            elif r["anl"]["spec"].get("high_freq"):
                pcfg=dict(pcfg); pcfg["arch"]="fourier"

            total_runs[0]+=1; rid=total_runs[0]
            lbl=r["label"]+"_deep"

            await send({"type":"run_start","gen":gen,"run":-1,"total":len(top_n),
                        "config":pcfg,"label":lbl,"run_id":rid,"phase":2})

            model=build_model(pde,pcfg.get("act","cos"),pcfg.get("width",128),
                               pcfg.get("depth",5),pcfg.get("arch","standard"),device,dtype)
            curr    = CurriculumScheduler()
            homo    = HomoscedasticWeighter(3,device=device,dtype=dtype)
            adp_w   = AdaptiveLossWeighter(3)
            dist    = SelfDistiller(teacher,alpha=0.25) if teacher is not None else None

            async def _cb2(d,_l=lbl):
                if d.get("type") in ("progress","status"): await send({**d,"label":_l,"phase":2})

            try:
                metrics,hist,elapsed=await _train(
                    model,pde,p2_ep,pcfg.get("lr",1e-3),n_col*2,_cb2,device,dtype,
                    solver=pcfg.get("solver","classic"),do_lbfgs=True,label=lbl,
                    curriculum=curr,distiller=dist,replay=replay,
                    homo_w=homo,adaptive_w=adp_w,meta=meta,
                    grad_ortho=grad_ortho,rg_sampler=rg_sampler,phase=2)
            except Exception as e:
                await send({"type":"run_error","gen":gen,"run":-1,"label":lbl,"msg":str(e)}); continue

            anl=_analyze(model,pde,device,dtype,lbl,pcfg,metrics,elapsed,rid,
                         bayes,failure,pareto,novelty,sigma_tuner,scorer,meta,
                         bandit,pop_ent,act_suit)

            # Update teacher if new best
            if best_ever is None or metrics["rel_l2"]<best_ever["metrics"]["rel_l2"]:
                teacher=copy.deepcopy(model)
                for p in teacher.parameters(): p.requires_grad_(False)
                teacher.eval()
            elif teacher is None:
                teacher=copy.deepcopy(model)
                for p in teacher.parameters(): p.requires_grad_(False)
                teacher.eval()

            update_hof(pcfg,metrics,elapsed,gen,rid,"deep",{
                "spec":anl["spec"],"ntk_kappa":anl["ntk_kappa"],
                "ntk_status":anl["ntk_status"],"novelty":anl["novelty"],"suitability":anl["suitability"]})

            if anl["compare"].get("pred"): compare_all["curves"][lbl]=anl["compare"]["pred"]
            fg=field_grid(model,pde,device,dtype)
            vis=vis_2d(model,pde,device,dtype)
            diag_t,diag_m=diagnose(hist,metrics)

            await send({
                "type":"deep_done","gen":gen,"label":lbl,"phase":2,
                "metrics":metrics,"elapsed":elapsed,
                "field_grid":fg,"field_3d":fg,"vis_data":vis,
                "compare_data":compare_all,"hall_of_fame":_hof_snap(hof),
                "pareto_front":pareto.front()[:5],
                "diag_title":diag_t,"diag_msg":diag_m,
                "spectral":anl["spec"],"ntk_kappa":anl["ntk_kappa"],
                "ntk_status":anl["ntk_status"],"ntk_advice":anl["ntk_advice"],
                "novelty":anl["novelty"],"suitability":anl["suitability"],
                "homo_state":homo.state(),"adaptive_state":adp_w.state(),
                "curriculum":curr.state(),"grad_ortho":grad_ortho.state(),
                "bayes":bayes.state(),"total_runs":total_runs[0],"runtime":rt(),
            })
            await asyncio.sleep(0)

        # ══════════════════════════════════════════════════════
        # CHAMPION — every 4 generations
        # ══════════════════════════════════════════════════════
        if gen>0 and gen%4==0 and best_ever:
            bc=dict(best_ever["cfg"])
            if best_ever.get("ntk_kappa",0)>1e6: bc["arch"]="resnet"; bc["depth"]=min(bc.get("depth",5),5)
            elif best_ever.get("spec",{}).get("high_freq"): bc["arch"]="fourier"
            total_runs[0]+=1
            await send({"type":"champion_start","gen":gen,"config":bc})

            model=build_model(pde,bc.get("act","cos"),bc.get("width",128),
                               bc.get("depth",5),bc.get("arch","standard"),device,dtype)
            curr=CurriculumScheduler()
            homo=HomoscedasticWeighter(3,device=device,dtype=dtype)
            adp_w=AdaptiveLossWeighter(3)

            async def _chcb(d):
                if d.get("type") in ("progress","status"): await send({**d,"label":"CHAMPION","phase":3})

            try:
                metrics,hist,elapsed=await _train(
                    model,pde,champ_ep,bc.get("lr",1e-3),n_col*3,_chcb,device,dtype,
                    solver="classic",do_lbfgs=True,label="CHAMPION",
                    curriculum=curr,replay=replay,homo_w=homo,adaptive_w=adp_w,
                    meta=meta,grad_ortho=grad_ortho,rg_sampler=rg_sampler,phase=3)
                anl=_analyze(model,pde,device,dtype,"CHAMPION",bc,metrics,elapsed,
                             total_runs[0],bayes,failure,pareto,novelty,sigma_tuner,
                             scorer,meta,bandit,pop_ent,act_suit)
                teacher=copy.deepcopy(model)
                for p in teacher.parameters(): p.requires_grad_(False)
                teacher.eval()
                update_hof(bc,metrics,elapsed,gen,total_runs[0],"champion",{
                    "spec":anl["spec"],"ntk_kappa":anl["ntk_kappa"],"novelty":anl["novelty"]})
                fg=field_grid(model,pde,device,dtype)
                vis=vis_2d(model,pde,device,dtype)
                await send({
                    "type":"champion_done","gen":gen,"metrics":metrics,"elapsed":elapsed,
                    "field_grid":fg,"field_3d":fg,"vis_data":vis,"compare_data":compare_all,
                    "hall_of_fame":_hof_snap(hof),"pareto_front":pareto.front()[:5],
                    "spectral":anl["spec"],"ntk_kappa":anl["ntk_kappa"],
                    "total_runs":total_runs[0],"runtime":rt(),
                })
            except Exception as e:
                await send({"type":"run_error","gen":gen,"run":-2,"label":"champion","msg":str(e)})

        await send({
            "type":"generation_done","gen":gen,"runtime":rt(),
            "total_runs":total_runs[0],
            "best_rl2":best_ever["metrics"]["rel_l2"] if best_ever else 1.0,
            "pareto_front":pareto.front(),
            "bayes":bayes.state(),"meta":meta.state(),"replay":replay.state(),
            "sigma":sigma_tuner.state(),"bandit":bandit.state(),
            "pop_entropy":pop_ent.state(),"grad_ortho":grad_ortho.state(),
            "spectral_hist":[s.get("dominant_freq",0) for s in spectral_hist[-10:]],
            "ntk_hist":[n for n in ntk_hist[-10:] if n>0],
        })
        await asyncio.sleep(0)

    await send({"type":"autonomous_done","total_runs":total_runs[0],"runtime":rt()})
