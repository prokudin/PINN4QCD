# PINN Research Platform

Autonomous self-training Physics-Informed Neural Network research platform.

## Features

- **4 Physics Problems** — Maxwell EM, Harmonic ODE, Burgers PDE, Heat Equation
- **20 Self-Training Algorithms** — Bayesian optimization, gradient surgery, NTK monitoring, spectral analysis, population entropy, meta-learning, and more
- **Custom Activation Functions** — Expression builder with live preview + derivative chart
- **3D Visualizations** — PDE-correct geometry per problem type:
  - Maxwell: E/B field ribbons in XY/XZ planes with Poynting vectors
  - Harmonic: Phase portrait (x × u × du/dx traces a circle)
  - Burgers/Heat: Spatiotemporal surface u(x,t) with shock/diffusion dynamics
- **Data Table** — Double-click any run to jump straight into its 3D visualization
- **Export** — PNG snapshots, field JSON/CSV, training data CSV

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/pinn-research
cd pinn-research
pip install -r requirements.txt
python run.py
```

Browser opens automatically at `http://localhost:8765`

## Project Structure

```
pinn-research/
├── run.py                  Entry point
├── requirements.txt
├── launch.sh               Unix launcher
├── launch.bat              Windows launcher
├── src/
│   ├── __init__.py
│   ├── physics.py          4 PDE loss functions + metrics + visualization data
│   ├── models.py           3 architectures × 18+ activations
│   ├── solvers.py          6 solver strategies + failure diagnostics
│   ├── algorithms.py       20 autonomous algorithms
│   ├── engine.py           Self-directing research loop
│   └── server.py           FastAPI + WebSocket + activation studio endpoints
└── static/
    └── index.html          Full frontend (no build step required)
```

## Physics Problems

| PDE | Domain | True Solution |
|-----|--------|---------------|
| Maxwell EM | (x,t)→(E,B) | E=B=sin(kx−ωt), c=k=ω=1 |
| Harmonic ODE | x→u | sin(x) |
| Burgers | (x,t)→u | Shock wave, ν=0.01/π |
| Heat Eq. | (x,t)→u | exp(−αt)·sin(x), α=0.1 |

## Architectures

| Name | Fixes |
|------|-------|
| Standard PINN | Baseline 5×128 cosine layers (SIREN init) |
| Fourier Feature | Spectral bias — high-frequency solutions |
| ResNet PINN | Vanishing gradients in deep networks |

## 20 Self-Training Algorithms

**Optimization:** BayesianOptimizer · ConfigScorer · MultiArmedBandit

**Gradient Management:** PCGrad · GradientOrthogonality · HomoscedasticWeighter · AdaptiveLossWeighter

**Adaptive Training:** CurriculumScheduler · ReplayBuffer · ResidualGuidedSampler

**Architecture Selection:** SpectralAnalyzer · FourierSigmaTuner · NTKMonitor · ActivationSuitability

**Population / Diversity:** ParetoTracker · NoveltySearch · FailureMemory · PopulationEntropy

**Knowledge Transfer:** MetaLearner · SelfDistiller

## Custom Activation Functions

### Math Expression Mode
Enter any PyTorch expression in `x`:
```
x * torch.sin(x)
torch.cos(5*x) * torch.exp(-0.5*x*x)
(1 - x*x) * torch.exp(-0.5*x*x)
```
Live preview, derivative chart, spectral suitability score, and gradient flow analysis.

### Parametric Mode (requires live backend)
Define a template with learnable parameters that get optimized for your PDE:
```
a * torch.sin(b * x) + c * x
```
The optimizer tunes `a, b, c` to maximize spectral overlap with the PDE solution.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/api/health` | GET | Device, PyTorch version, available acts/PDEs |
| `/api/validate_act` | POST | Compile + preview custom expression |
| `/api/act_gradflow` | POST | Gradient flow analysis |
| `/api/act_suitability` | POST | Spectral suitability vs PDE |
| `/api/act_parametric` | POST | Optimize parametric activation |
| `/api/act_benchmark` | POST | Benchmark vs preset activations |
| `/ws` | WebSocket | Training stream (single/search/sweep/autonomous) |

## WebSocket Protocol

```json
{
  "mode": "single" | "search" | "sweep" | "autonomous",
  "pde": "maxwell" | "harmonic" | "burgers" | "heat",
  "suite": "cos" | "sin" | "tanh" | ... | "custom",
  "custom_act_expr": "x * torch.sin(x)",
  "epochs": 2000,
  "width": 128,
  "depth": 5,
  "lr": 0.001,
  "n_colloc": 2048,
  "use_lbfgs": true
}
```

## Requirements

- Python 3.9+
- PyTorch 2.0+ (CPU works; CUDA/MPS auto-detected)
- FastAPI + Uvicorn
- NumPy

No GPU required.
