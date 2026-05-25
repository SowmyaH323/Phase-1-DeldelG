# Phase 1 ΔΔG Mutation Scanner — Weighted Ensemble

This folder contains a Streamlit deployment version of the Phase 1 ΔΔG prediction tool.

## Ensemble weights

The app uses the manuscript weighted ensemble:

- XGB = 0.45
- CNN = 0.35
- GNN = 0.20

If only FASTA/sequence is supplied, the tool uses XGB only. If a PDB file is supplied and CNN/GNN models load correctly, the weighted ensemble is calculated. If one structural model fails, the available model weights are re-normalized.

## Required model files

Place these files inside the `models/` folder:

```text
models/xgb_full.json
models/cnn_ddg_model.keras
models/gnn_esm_scripted.pt
```

The empty `models/` folder is included here only as a placeholder. Large model files must be uploaded separately.

## Files

```text
app.py
requirements.txt
README.md
models/
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit Cloud

Upload the complete folder contents to GitHub. In Streamlit Cloud, set:

```text
Main file path: app.py
```
