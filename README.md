# TEMPLATE-RL-Projects

A lightweight template to bootstrap new RL visual projects inspired by `RL-Light-Chaser`.

Use this as a starting point to build small interactive RL demos (e.g., multi‑armed bandits, simple environments) with a clean folder layout and a minimal UI scaffold.

## Structure
```
TEMPLATE-RL-Projects/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── requirements-312.txt
├── notebooks/
│   └── (your notebooks)
├── assets/
│   └── images/            # put background/buttons here (kept empty in template)
├── config/
│   └── project.json       # example config
├── scripts/
│   └── demo_template.py   # example entry-point
└── src/
    ├── agents/            # RL logic (policies, controllers)
    ├── comms/             # IPC/bridges (optional)
    ├── engine/            # rendering/UX (Arcade)
    ├── envs/              # simple envs/simulators (optional)
    ├── physics/           # dynamics/physics helpers (optional)
    ├── simulation/        # experiment runners/batching (optional)
    ├── utils/             # helpers (config loader, logging)
    └── viz/               # plots/overlays (optional)
```

## Quick start (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Python 3.10–3.11
pip install -r requirements.txt
# or Python 3.12
pip install -r requirements-312.txt

python -m scripts.demo_template
```

## Configuration
Edit `config/project.json` to set initial parameters for your demo (e.g., RNG seed, UI toggles). This file is intentionally minimal.

## Notes
- This template does NOT include art assets or advanced logic; it is a clean slate with a working window loop built on `arcade`.
- Copy this folder and rename it to start a new project.
- License defaults to The Unlicense (public domain dedication); change as needed.
