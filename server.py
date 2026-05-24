"""
server.py v5 — FastAPI + WebSocket + Activation Studio endpoints
=================================================================
New in v5:
  POST /api/validate_act       compile + preview custom expression
  POST /api/act_gradflow       gradient flow analysis of custom act
  POST /api/act_suitability    spectral suitability vs PDE
  POST /api/act_evolve         one generation of ActivationEvolver
  POST /api/act_parametric     optimize parametric activation
  POST /api/act_benchmark      benchmark custom act vs presets
  GET  /api/algorithms         current state of all algorithm instances
"""

import json, traceback, pathlib, copy, time, asyncio, math
import torch, torch.nn.functional as F

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .physics   import (compute_metrics, vis_2d, field_grid, compare_curve,
                         compute_total_loss, PDE_REGISTRY)
from .models    import build_model, ACTIVATIONS
from .solvers   import run_solver, diagnose, SOLVER_INFO
from .engine    import stream_autonomous
from .algorithms import (
    ActivationSuitability, ActivationEvolver,
    ActivationGradFlowAnalyzer, ActivationBenchmark,
    ParametricActivationOptimizer,
)

# ── Device ─────────────────────────────────────────────────
def _dev():
    if torch.cuda.is_available():    return torch.device("cuda")
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _dev(); DTYPE = torch.float64
torch.set_default_dtype(torch.float64)
_HERE = pathlib.Path(__file__).parent.parent
_IDX  = _HERE / "static" / "index.html"
app   = FastAPI(title="PINN Research Platform v5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Shared evolver instance (persists across requests this session)
_evolvers: dict = {}

# ── Custom activation compiler ──────────────────────────────
_SAFE_NS = {
    "torch": torch, "math": math, "F": F,
    "sin": torch.sin, "cos": torch.cos, "tanh": torch.tanh,
    "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt,
    "abs": torch.abs, "relu": torch.relu, "sigmoid": torch.sigmoid,
    "softplus": F.softplus, "erf": torch.erf, "selu": F.selu,
    "swish": lambda x: x * torch.sigmoid(x),
    "__builtins__": {},
}

def compile_act(expr: str):
    expr = expr.strip()
    if not expr: raise ValueError("Empty expression")
    for banned in ("import", "os.", "sys.", "open(", "exec(", "eval(", "__import__"):
        if banned in expr: raise ValueError(f"Disallowed token: {banned!r}")
    code  = compile(f"lambda x: {expr}", "<act>", "eval")
    fn    = eval(code, dict(_SAFE_NS))
    with torch.no_grad():
        y = fn(torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0], dtype=torch.float64))
        if not torch.isfinite(y).all(): raise ValueError("Expression produces NaN/Inf at test points")
    return fn

def reg_custom(cfg: dict):
    expr = cfg.get("custom_act_expr", "").strip()
    if not expr: return None
    fn = compile_act(expr)
    ACTIVATIONS["custom"] = fn
    return "custom"

# ── Routes ──────────────────────────────────────────────────
@app.get("/")
async def root(): return HTMLResponse(_IDX.read_text(encoding="utf-8"))

@app.get("/api/health")
async def health():
    return JSONResponse({
        "status":  "ok", "device": str(DEVICE), "torch": torch.__version__,
        "acts":    list(ACTIVATIONS.keys()), "pdes": list(PDE_REGISTRY.keys()),
        "solvers": list(SOLVER_INFO.keys()),
        "cuda":    torch.cuda.is_available(),
        "mps":     bool(getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()),
        "n_algos": 25,
    })

# ── Activation Studio endpoints ─────────────────────────────

@app.post("/api/validate_act")
async def validate_act(body: dict):
    """Compile expression and return preview + basic stats."""
    expr = body.get("expr", "")
    try:
        fn = compile_act(expr)
        with torch.no_grad():
            xs  = torch.linspace(-4.0, 4.0, 80, dtype=torch.float64)
            ys  = fn(xs)
            if not torch.isfinite(ys).all():
                return {"ok": False, "error": "Produces NaN/Inf on [-4,4]"}
            # Derivative preview
            xd  = xs.clone().requires_grad_(True)
            yd  = fn(xd)
            dyd = torch.autograd.grad(yd.sum(), xd)[0]
        return {
            "ok":     True,
            "x":      xs.tolist(),
            "y":      ys.tolist(),
            "dy":     dyd.detach().tolist(),
            "f0":     round(float(fn(torch.tensor([0.0], dtype=torch.float64))), 5),
            "f1":     round(float(fn(torch.tensor([1.0], dtype=torch.float64))), 5),
            "max":    round(float(ys.max()), 4),
            "min":    round(float(ys.min()), 4),
            "msg":    f"Valid  f(0)={float(fn(torch.tensor([0.0],dtype=torch.float64))):.4f}  f(1)={float(fn(torch.tensor([1.0],dtype=torch.float64))):.4f}",
        }
    except ValueError as e: return {"ok": False, "error": str(e)}
    except Exception as e:   return {"ok": False, "error": f"Compile error: {e}"}


@app.post("/api/act_gradflow")
async def act_gradflow(body: dict):
    """Gradient flow analysis through test network."""
    expr   = body.get("expr", "")
    depth  = int(body.get("depth", 6))
    in_dim = int(body.get("in_dim", 2))
    try:
        fn = compile_act(expr)
        ana = ActivationGradFlowAnalyzer(test_depth=depth, test_width=64)
        return {"ok": True, "analysis": ana.analyze(fn, in_dim=in_dim)}
    except Exception as e: return {"ok": False, "error": str(e)}


@app.post("/api/act_suitability")
async def act_suitability(body: dict):
    """Spectral suitability score vs PDE solution."""
    expr = body.get("expr", "")
    pde  = body.get("pde", "maxwell")
    try:
        fn    = compile_act(expr)
        score = ActivationSuitability.score(fn, pde)
        # Also score against all presets for comparison
        preset_scores = {}
        for name, act in ACTIVATIONS.items():
            if name == "custom": continue
            try: preset_scores[name] = round(ActivationSuitability.score(act, pde), 4)
            except: pass
        ranked = sorted(preset_scores.items(), key=lambda x: -x[1])
        rank   = sum(1 for _, v in ranked if v > score) + 1
        return {
            "ok":     True,
            "score":  round(score, 4),
            "rank":   rank,
            "total":  len(ranked) + 1,
            "ranked": [{"act": k, "score": v} for k, v in ranked[:8]],
        }
    except Exception as e: return {"ok": False, "error": str(e)}


@app.post("/api/act_evolve")
async def act_evolve(body: dict):
    """Run one generation of ActivationEvolver."""
    pde      = body.get("pde", "maxwell")
    n_gens   = int(body.get("n_gens", 5))
    seed_expr = body.get("seed_expr", "")
    key = pde

    if key not in _evolvers:
        _evolvers[key] = ActivationEvolver(pde=pde, pop_size=16)

    ev = _evolvers[key]

    # Inject user seed if provided
    if seed_expr:
        fn = None
        try: fn = compile_act(seed_expr)
        except: pass
        if fn:
            score = ActivationSuitability.score(fn, pde)
            ev.population.insert(0, {"expr": seed_expr, "score": score, "gen": ev.generation})

    results = []
    for _ in range(n_gens):
        r = ev.step()
        results.append(r)

    return {"ok": True, "state": ev.state(), "history": results}


@app.post("/api/act_parametric")
async def act_parametric(body: dict):
    """Optimize a parametric activation expression."""
    template    = body.get("template", "a * torch.sin(b * x)")
    param_names = body.get("param_names", ["a", "b"])
    bounds      = body.get("bounds", {p: [0.1, 3.0] for p in param_names})
    pde         = body.get("pde", "maxwell")
    n_iters     = int(body.get("n_iters", 60))

    bounds_dict = {k: tuple(v) for k, v in bounds.items()}
    try:
        opt    = ParametricActivationOptimizer(template, param_names, bounds_dict)
        result = opt.optimize(pde=pde, n_iters=n_iters)
        return {"ok": True, "result": result}
    except Exception as e: return {"ok": False, "error": str(e)}


@app.post("/api/act_benchmark")
async def act_benchmark(body: dict):
    """Rank custom activation against presets using spectral suitability."""
    expr    = body.get("expr", "")
    pde     = body.get("pde", "maxwell")
    presets = body.get("presets", ["cos","sin","tanh","swish","gelu","morlet"])
    name    = body.get("name", "custom")
    try:
        fn    = compile_act(expr)
        bench = ActivationBenchmark()
        result = bench.run_async_friendly(fn, name, presets, pde)
        return {"ok": True, "result": result}
    except Exception as e: return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
# WEBSOCKET MODES
# ══════════════════════════════════════════════════════════

async def _single(ws, cfg):
    ck  = reg_custom(cfg)
    pde = cfg.get("pde","harmonic")
    act = ck or cfg.get("suite", cfg.get("act","cos"))
    if act not in ACTIVATIONS: act="cos"
    epochs=int(cfg.get("epochs",2000)); width=int(cfg.get("width",128))
    depth=int(cfg.get("depth",5));      lr=float(cfg.get("lr",1e-3))
    n_col=int(cfg.get("n_colloc",2048)); lbfgs=bool(cfg.get("use_lbfgs",True))
    solver=cfg.get("solver","classic");  arch=cfg.get("arch","standard")
    lbl = cfg.get("custom_act_expr",act) if ck else act

    async def send(d):
        try: await ws.send_text(json.dumps(d))
        except: pass

    await send({"type":"status","msg":f"{DEVICE} | {pde} | {lbl} | {arch} | {solver}"})
    model = build_model(pde, act, width, depth, arch, DEVICE, DTYPE)
    await send({"type":"status","msg":f"{model.n_params():,} parameters"})
    metrics,hist,elapsed = await run_solver(
        model,pde,epochs,lr,n_col,send,DEVICE,DTYPE,
        solver=solver,do_lbfgs=lbfgs,label=lbl)
    dt,dm = diagnose(hist,metrics)
    fg  = field_grid(model,pde,DEVICE,DTYPE)
    vis = vis_2d(model,pde,DEVICE,DTYPE)
    cmp = compare_curve(model,pde,DEVICE,DTYPE)
    await send({"type":"done","ratio":metrics["ratio"],"pct":metrics["pct"],
                "rel_l2":metrics["rel_l2"],"elapsed":round(elapsed,1),
                "n_params":model.n_params(),"act":lbl,
                "vis_data":vis,"field_grid":fg,"field_3d":fg,
                "compare":cmp,"history":hist,"diag_title":dt,"diag_msg":dm})


async def _search(ws, cfg):
    ck   = reg_custom(cfg)
    acts = [a for a in cfg.get("activations",["cos","tanh"]) if a in ACTIVATIONS]
    if ck and ck not in acts: acts.insert(0,ck)
    pde=cfg.get("pde","harmonic"); epochs=int(cfg.get("epochs",1000))
    width=int(cfg.get("width",128)); depth=int(cfg.get("depth",5))
    lr=float(cfg.get("lr",1e-3)); n_col=int(cfg.get("n_colloc",2048))
    compare_data={"x":None,"true":None,"curves":{}}; leaderboard=[]

    async def send(d):
        try: await ws.send_text(json.dumps(d))
        except: pass

    await send({"type":"status","msg":f"Search: {len(acts)} acts | {pde} | {epochs}ep each"})
    for idx,act in enumerate(acts):
        lbl = cfg.get("custom_act_expr",act) if act=="custom" else act
        await send({"type":"search_start","act":lbl,"index":idx,"total":len(acts)})
        model = build_model(pde,act,width,depth,"standard",DEVICE,DTYPE)
        async def pg(d,_l=lbl,_i=idx):
            if d.get("type")=="progress":
                await send({"type":"search_progress","act":_l,"index":_i,
                            "epoch":d["epoch"],"n_epochs":d["n_epochs"],
                            "ratio":d["ratio"],"rel_l2":d["rel_l2"],"elapsed":d["elapsed"]})
        try:
            metrics,hist,elapsed = await run_solver(
                model,pde,epochs,lr,n_col,pg,DEVICE,DTYPE,
                solver="classic",do_lbfgs=True,label=lbl)
        except Exception as e:
            await send({"type":"error","msg":f"{lbl}: {e}"}); continue
        cmp = compare_curve(model,pde,DEVICE,DTYPE)
        if compare_data["x"] is None:
            compare_data["x"]=cmp.get("x",[]); compare_data["true"]=cmp.get("true",cmp.get("E_true",[]))
        compare_data["curves"][lbl]=cmp.get("pred",cmp.get("E_pred",[]))
        fg  = field_grid(model,pde,DEVICE,DTYPE) if pde=="maxwell" else None
        vis = vis_2d(model,pde,DEVICE,DTYPE)
        result={"act":lbl,"ratio":metrics["ratio"],"pct":metrics["pct"],
                "rel_l2":metrics["rel_l2"],"elapsed":round(elapsed,1),"n_params":model.n_params()}
        leaderboard.append(result); leaderboard.sort(key=lambda x:x["rel_l2"])
        for j,r in enumerate(leaderboard): r["rank"]=j+1
        await send({"type":"search_result","act":lbl,"index":idx,"total":len(acts),
                    "result":result,"leaderboard":leaderboard,"compare_data":compare_data,
                    "vis_data":vis,"field_grid":fg,"field_3d":fg})
        await asyncio.sleep(0)
    await send({"type":"search_done","leaderboard":leaderboard,"compare_data":compare_data,
                "pde":pde,"winner":leaderboard[0] if leaderboard else None})


async def _sweep(ws, cfg):
    ck   = reg_custom(cfg)
    acts = [a for a in cfg.get("activations",["cos","tanh","swish"]) if a in ACTIVATIONS]
    if ck and ck not in acts: acts.insert(0,ck)
    pde=cfg.get("pde","harmonic"); widths=[int(w) for w in cfg.get("widths",[64,128])]
    depths=[int(d) for d in cfg.get("depths",[3,5])]; lrs=[float(l) for l in cfg.get("lrs",[1e-3,3e-4])]
    ep1=int(cfg.get("phase1_epochs",400)); ep2=int(cfg.get("phase2_epochs",2000))
    top_k=int(cfg.get("top_k",3)); n_col=int(cfg.get("n_colloc",2048)); seed=int(cfg.get("seed",42))

    async def send(d):
        try: await ws.send_text(json.dumps(d))
        except: pass

    configs=[{"act":a,"width":w,"depth":d,"lr":l,
              "act_label":(cfg.get("custom_act_expr",a) if a=="custom" else a),
              "cfg_id":f"{(cfg.get('custom_act_expr',a) if a=='custom' else a)}|w{w}d{d}|lr{l:.0e}"}
             for a in acts for w in widths for d in depths for l in lrs]
    await send({"type":"sweep_plan","total":len(configs),"pde":pde,
                "phase1_epochs":ep1,"phase2_epochs":ep2,"top_k":top_k})
    compare_data={"x":None,"true":None,"curves":{}}; p1_results=[]
    await send({"type":"phase","phase":1,"label":f"Rapid scan — {len(configs)} × {ep1}ep"})

    for idx,c in enumerate(configs):
        await send({"type":"sweep_start","index":idx,"total":len(configs),"cfg_id":c["cfg_id"],"phase":1})
        torch.manual_seed(seed)
        model=build_model(pde,c["act"],c["width"],c["depth"],"standard",DEVICE,DTYPE)
        async def p1p(d,_c=c,_i=idx):
            if d.get("type")=="progress":
                await send({"type":"sweep_progress","index":_i,"total":len(configs),
                            "cfg_id":_c["cfg_id"],"pct":round(d["epoch"]/d["n_epochs"]*100),
                            "phase":1,"ratio":d["ratio"],"rel_l2":d["rel_l2"]})
        try:
            metrics,hist,elapsed=await run_solver(model,pde,ep1,c["lr"],n_col,p1p,
                DEVICE,DTYPE,solver="classic",do_lbfgs=False,label=c["cfg_id"])
        except Exception as e:
            await send({"type":"error","msg":f"P1 {c['cfg_id']}: {e}"}); continue
        cmp=compare_curve(model,pde,DEVICE,DTYPE)
        if compare_data["x"] is None:
            compare_data["x"]=cmp.get("x",[]); compare_data["true"]=cmp.get("true",cmp.get("E_true",[]))
        compare_data["curves"][c["act_label"]]=cmp.get("pred",cmp.get("E_pred",[]))
        result={**c,"ratio":metrics["ratio"],"pct":metrics["pct"],"rel_l2":metrics["rel_l2"],
                "elapsed":round(elapsed,1),"phase":1}
        p1_results.append(result); p1_results.sort(key=lambda x:x["rel_l2"])
        for j,r in enumerate(p1_results): r["rank"]=j+1
        await send({"type":"sweep_result","result":result,"phase":1,
                    "leaderboard":p1_results[:20],"compare_data":copy.deepcopy(compare_data)})
        await asyncio.sleep(0)

    await send({"type":"phase_done","phase":1,
                "winner_p1":p1_results[0]["cfg_id"] if p1_results else "—",
                "leaderboard_p1":p1_results[:20]})
    top_configs=p1_results[:min(top_k,len(p1_results))]
    await send({"type":"phase","phase":2,"label":f"Deep retrain — top {len(top_configs)} × {ep2}ep + L-BFGS"})
    p2_results=[]

    for idx,c in enumerate(top_configs):
        await send({"type":"sweep_start","index":idx,"total":len(top_configs),"cfg_id":c["cfg_id"],"phase":2})
        torch.manual_seed(seed)
        model=build_model(pde,c["act"],c["width"],c["depth"],"standard",DEVICE,DTYPE)
        async def p2p(d,_c=c,_i=idx):
            if d.get("type")=="progress":
                await send({"type":"sweep_progress","index":_i,"total":len(top_configs),
                            "cfg_id":_c["cfg_id"],"pct":round(d["epoch"]/d["n_epochs"]*100),
                            "phase":2,"ratio":d["ratio"],"rel_l2":d["rel_l2"]})
        try:
            metrics,hist,elapsed=await run_solver(model,pde,ep2,c["lr"],n_col*2,p2p,
                DEVICE,DTYPE,solver="classic",do_lbfgs=True,label=c["cfg_id"])
        except Exception as e:
            await send({"type":"error","msg":f"P2 {c['cfg_id']}: {e}"}); continue
        fg=field_grid(model,pde,DEVICE,DTYPE) if pde=="maxwell" else None
        vis=vis_2d(model,pde,DEVICE,DTYPE); cmp=compare_curve(model,pde,DEVICE,DTYPE)
        compare_data["curves"][c["act_label"]+"_p2"]=cmp.get("pred",cmp.get("E_pred",[]))
        result={**c,"ratio":metrics["ratio"],"pct":metrics["pct"],"rel_l2":metrics["rel_l2"],
                "elapsed":round(elapsed,1),"phase":2,"vis_data":vis,"field_grid":fg,"field_3d":fg}
        p2_results.append(result); p2_results.sort(key=lambda x:x["rel_l2"])
        for j,r in enumerate(p2_results): r["rank"]=j+1
        await send({"type":"sweep_result","result":result,"phase":2,"leaderboard_p2":p2_results,
                    "vis_data":vis,"field_grid":fg,"field_3d":fg,"compare_data":copy.deepcopy(compare_data)})
        await asyncio.sleep(0)

    winner=p2_results[0] if p2_results else (p1_results[0] if p1_results else None)
    await send({"type":"sweep_done","leaderboard_p1":p1_results[:20],"leaderboard_p2":p2_results,
                "winner":winner,"field_3d":winner.get("field_grid") if winner else None,
                "compare_data":compare_data,"pde":pde})


@app.websocket("/ws")
async def ws_main(ws: WebSocket):
    await ws.accept()
    try:
        raw=await ws.receive_text(); cfg=json.loads(raw)
        mode=cfg.get("mode","single")
        dispatch={"single":  lambda: _single(ws,cfg),
                  "search":  lambda: _search(ws,cfg),
                  "sweep":   lambda: _sweep(ws,cfg),
                  "autonomous": lambda: stream_autonomous(ws,cfg,DEVICE,DTYPE)}
        h=dispatch.get(mode)
        if h: await h()
        else: await ws.send_text(json.dumps({"type":"error","msg":f"Unknown mode: {mode!r}"}))
    except WebSocketDisconnect: pass
    except Exception as e:
        try: await ws.send_text(json.dumps({"type":"error","msg":str(e),"tb":traceback.format_exc()}))
        except: pass
    finally:
        try: await ws.close()
        except: pass


# ══════════════════════════════════════════════════════════
# PYTHON REPL ENDPOINT
# ══════════════════════════════════════════════════════════

import sys, io, contextlib, traceback as _tb, threading

_repl_globals = {
    "torch":   torch,
    "np":      __import__("numpy"),
    "math":    math,
    "DEVICE":  None,   # filled below
    "DTYPE":   None,
    "ACTIVATIONS": None,
    "build_model": None,
    "compute_metrics": None,
    "field_grid": None,
    "vis_2d":   None,
    "print":    print,
}

def _init_repl():
    from .models  import build_model as _bm, ACTIVATIONS as _A
    from .physics import compute_metrics as _cm, field_grid as _fg, vis_2d as _v2
    _repl_globals.update({
        "DEVICE": DEVICE, "DTYPE": DTYPE,
        "build_model": _bm, "ACTIVATIONS": _A,
        "compute_metrics": _cm, "field_grid": _fg, "vis_2d": _v2,
    })

_repl_lock = threading.Lock()
_repl_session: dict = {}   # cell_id → result

@app.post("/api/exec")
async def exec_code(body: dict):
    """Execute arbitrary Python code in a sandboxed REPL session."""
    code    = body.get("code", "")
    cell_id = body.get("cell_id", "")
    reset   = body.get("reset", False)

    if reset:
        _repl_session.clear()
        _init_repl()
        return {"ok": True, "output": "Session reset.", "cell_id": cell_id}

    if not code.strip():
        return {"ok": True, "output": "", "cell_id": cell_id}

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    result_val = None

    with _repl_lock:
        try:
            _init_repl()
            # Try expression first (returns value)
            try:
                code_obj = compile(code, "<repl>", "eval")
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    result_val = eval(code_obj, _repl_globals)
            except SyntaxError:
                # Fall back to exec for statements
                code_obj = compile(code, "<repl>", "exec")
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(code_obj, _repl_globals)

            out = stdout_buf.getvalue()
            err = stderr_buf.getvalue()

            output = out
            if err:   output += ("\n" if output else "") + err
            if result_val is not None:
                output += ("\n" if output else "") + repr(result_val)

            _repl_session[cell_id] = {"ok": True, "output": output.rstrip()}
            return {"ok": True, "output": output.rstrip(), "cell_id": cell_id}

        except Exception as e:
            tb = _tb.format_exc()
            # Trim to last 12 lines of traceback
            tb_lines = tb.strip().split("\n")
            short_tb = "\n".join(tb_lines[-12:])
            _repl_session[cell_id] = {"ok": False, "output": short_tb}
            return {"ok": False, "output": short_tb, "cell_id": cell_id}


@app.get("/api/exec/history")
async def exec_history():
    return JSONResponse({"history": list(_repl_session.items())[-50:]})


# ══════════════════════════════════════════════════════════
# SNAPSHOT / EXPORT ENDPOINT
# ══════════════════════════════════════════════════════════

import base64, json as _json

@app.post("/api/snapshot_meta")
async def snapshot_meta(body: dict):
    """Accept field_grid JSON from frontend, return enriched metadata for download."""
    fg = body.get("field_grid")
    if not fg:
        return {"ok": False, "error": "No field_grid provided"}

    try:
        meta = {
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device":      str(DEVICE),
            "torch":       torch.__version__,
            "pde":         fg.get("pde","maxwell"),
            "nx":          fg.get("nx", len(fg.get("x",[]))),
            "nt":          fg.get("nt", len(fg.get("t",[]))),
            "x_range":     [fg["x"][0], fg["x"][-1]] if fg.get("x") else [],
            "t_range":     [fg["t"][0], fg["t"][-1]] if fg.get("t") else [],
            "n_points":    fg.get("nx",0) * fg.get("nt",0),
        }
        # Add error statistics if both PINN and true data present
        if fg.get("E_pred") and fg.get("E_true"):
            Ep = torch.tensor(fg["E_pred"], dtype=torch.float64)
            Et = torch.tensor(fg["E_true"], dtype=torch.float64)
            err = (Ep - Et).abs()
            meta["E_error"] = {
                "mean":  round(float(err.mean()), 6),
                "max":   round(float(err.max()), 6),
                "rmse":  round(float((err**2).mean().sqrt()), 6),
                "rel_l2":round(float(err.norm()/(Et.norm()+1e-8)),6),
            }
        return {"ok": True, "meta": meta}
    except Exception as e:
        return {"ok": False, "error": str(e)}
