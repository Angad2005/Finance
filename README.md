# Portfolio RL Experiment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## Overview

This repository provides a **reproducible experimental pipeline** that evaluates an **LSTM‑based reinforcement‑learning (RL) portfolio agent** against three classical baselines:

- **UBH** – Uniform Buy‑and‑Hold (simple equal‑weight benchmark)
- **MVO** – Mean‑Variance Optimization (Markowitz portfolio)
- **MLP‑RL** – Multi‑Layer Perceptron policy trained with PPO

The evaluation follows a **Walk‑Forward Validation (WFV)** scheme to mimic realistic out‑of‑sample trading periods. All code is container‑friendly and can be run on a standard laptop with a GPU.

---

## Repository Structure

```
portfolio_rl_experiment/
├─ .gitignore          # ignored files (virtual env, caches, data)
├─ LICENSE             # MIT license
├─ README.md           # this file
├─ requirements.txt   # pinned Python dependencies
├─ config.yaml         # experiment configuration
├─ check_versions.py   # sanity‑check of package versions
├─ run_experiment.py   # end‑to‑end entry point
├─ run_ablation.py    # optional ablation script for plots
├─ data/               # (optional) local data cache
├─ results/           # generated metrics, plots, models, logs
└─ src/                # source code
   ├─ __init__.py
   ├─ env_portfolio.py
   ├─ lstm_rl_agent.py
   ├─ baselines.py
   ├─ backtest_runner.py
   ├─ metrics.py
   ├─ stats_test.py
   └─ ...
```

---

## Setup and Execution

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/portfolio_rl_experiment.git
   cd portfolio_rl_experiment
   ```

2. **Create a virtual environment and activate it**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the strictly pinned requirements**
   ```bash
   pip install -r requirements.txt
   ```
   > If you encounter wheel conflicts, you can add `--prefer-binary` or use `pip install --no-cache-dir`.

4. **Verify the installed package versions** (optional sanity check)
   ```bash
   python check_versions.py
   ```

5. **Run the full experiment**
   ```bash
   python run_experiment.py --config config.yaml
   ```
   The script will:
   - Download (or synthesize) price data via `yfinance`.
   - Perform walk‑forward splits.
   - Train the three RL agents (MLP‑RL, LSTM‑RL) and compute the baselines.
   - Save metrics, plots, models, and a comprehensive log under `results/`.

---

## Results

All outputs are saved to the `results/` directory, organized as follows:

- `metrics/performance_summary.csv` – tabular performance (cumulative return, Sharpe, max‑drawdown, turnover).
- `metrics/stat_tests.json` – Diebold‑Mariano test statistics and bootstrap confidence intervals.
- `plots/` – PNGs of equity curves, rolling Sharpe, and LSTM weight allocation over time.
- `models/` – serialized SB3 agents for reproducibility.
- `experiment.log` – detailed run‑time log (training progress, warnings, timestamps).

The repository includes a tiny helper script `results/table.tex` that converts the CSV summary into a LaTeX `tabular` environment, ready for inclusion in a research paper.

---

## Reproducibility

- The experiment seeds are fixed (`experiment.seed = 42`) and deterministic PyTorch flags are set, guaranteeing bit‑wise reproducibility on the same hardware.
- The configuration file `config.yaml` contains all hyper‑parameters (learning rates, total timesteps, transaction cost, etc.). Adjusting any of these values and re‑running `run_experiment.py` will produce a new, self‑contained results folder.
- The code is version‑controlled; you can tag a specific commit to reference the exact code used for a paper.

---

## Contribution

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your‑feature`).
3. Make your changes, ensuring the test suite (if added) passes.
4. Submit a Pull Request with a clear description of the changes.

Feel free to open an issue for any bug reports or feature requests.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## Acknowledgements

The implementation builds upon:
- **Stable‑Baselines3** for PPO agents.
- **Gymnasium** for the custom portfolio environment.
- **yfinance** for data ingestion.
- Various finance research papers for metric definitions and statistical tests.

---

*Happy experimenting!*

This repository contains a complete, reproducible experimental pipeline that compares an LSTM-based RL portfolio agent against three baselines (Uniform Buy-and-Hold, Mean-Variance Optimization, MLP-RL) using Walk-Forward Validation (WFV).

## Disclaimer
Yahoo Finance data is for research purposes only. Commercial redistribution is strictly prohibited per their terms of service.

## Setup and Execution

1. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the strictly pinned requirements:
```bash
pip install -r requirements.txt
```
*(If you encounter wheel conflicts, append `--prefer-binary` or use `pip install --no-cache-dir`)*

3. Verify the installed package versions:
```bash
python check_versions.py
```

4. Run the experiment:
```bash
python run_experiment.py --config config.yaml
```

## Results

All outputs will be saved to the `results/` directory, including:
- Metrics summaries (CSV/JSON)
- Plots (Equity curves, rolling Sharpe, LSTM weight allocation)
- Saved models and weights
- Comprehensive logs
