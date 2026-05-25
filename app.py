# app.py — Phase 1 ΔΔG Mutation Scanner
# Weighted ensemble: XGB = 0.45, CNN = 0.35, GNN = 0.20
# FASTA-only mode: XGB prediction
# FASTA + PDB mode: XGB + CNN + GNN weighted ensemble when models are available

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# Optional imports are loaded lazily inside functions to avoid startup failure
# if TensorFlow/Torch model loading has an issue.

st.set_page_config(
    page_title="ΔΔG Mutation Scanner",
    layout="wide"
)

# -----------------------------
# Paths and constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

XGB_PATH = os.path.join(MODEL_DIR, "xgb_full.json")
CNN_PATH = os.path.join(MODEL_DIR, "cnn_ddg_model.keras")
GNN_PATH = os.path.join(MODEL_DIR, "gnn_esm_scripted.pt")

ENSEMBLE_WEIGHTS = {
    "xgb": 0.45,
    "cnn": 0.35,
    "gnn": 0.20
}

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA2IDX = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydrophobicity scale
_HYDRO = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
}

# Approximate residue side-chain volume scale used in the original app
_VOL = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0
}

_CHG = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}

CNN_INPUT_SIZE = 128


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_xgb_model():
    if not os.path.exists(XGB_PATH):
        return None, f"XGB model not found at {XGB_PATH}"
    try:
        booster = xgb.Booster()
        booster.load_model(XGB_PATH)
        return booster, None
    except Exception as e:
        return None, f"XGB loading failed: {e}"


@st.cache_resource(show_spinner=False)
def load_cnn_model():
    if not os.path.exists(CNN_PATH):
        return None, f"CNN model not found at {CNN_PATH}"
    try:
        from tensorflow.keras.models import load_model
        model = load_model(CNN_PATH)
        return model, None
    except Exception as e:
        return None, f"CNN loading failed: {e}"


@st.cache_resource(show_spinner=False)
def load_gnn_model():
    if not os.path.exists(GNN_PATH):
        return None, "GNN model not found"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.jit.load(GNN_PATH, map_location=device)
        model.eval()
        return (model, device), None
    except Exception as e:
        return None, f"GNN loading failed: {e}"


# -----------------------------
# Input parsers
# -----------------------------
def parse_fasta(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[0].startswith(">"):
        lines = lines[1:]
    seq = "".join(lines).upper().replace(" ", "")
    return "".join([c for c in seq if c in AA])


def parse_mutations(mut_str: str):
    """
    Accepts mutation input such as:
    I38D
    I38D;A42V
    I38D, A42V
    Returns list of dictionaries.
    """
    out = []
    if not mut_str:
        return out

    for tok in mut_str.replace(",", ";").split(";"):
        tok = tok.strip().upper()
        if len(tok) < 3:
            continue

        i = 1
        while i < len(tok) and tok[i].isdigit():
            i += 1

        if tok[0] in AA and i < len(tok) and tok[i] in AA:
            try:
                wt = tok[0]
                pos = int(tok[1:i])
                mt = tok[i]
                out.append({"wt": wt, "position": pos, "mt": mt, "mutation": f"{wt}{pos}{mt}"})
            except ValueError:
                pass
    return out


# -----------------------------
# Feature builders
# -----------------------------
def build_xgb_features(rows: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    df["position"] = df["position"].astype(float)
    df["wt_idx"] = df["wt"].map(AA2IDX).astype(float)
    df["mt_idx"] = df["mt"].map(AA2IDX).astype(float)
    df["d_hydro"] = df.apply(lambda r: _HYDRO.get(r["mt"], 0.0) - _HYDRO.get(r["wt"], 0.0), axis=1)
    df["d_volume"] = df.apply(lambda r: _VOL.get(r["mt"], 0.0) - _VOL.get(r["wt"], 0.0), axis=1)
    df["d_charge"] = df.apply(lambda r: _CHG.get(r["mt"], 0.0) - _CHG.get(r["wt"], 0.0), axis=1)

    return df[["position", "wt_idx", "mt_idx", "d_hydro", "d_volume", "d_charge"]]


def pdb_to_ca_coords(pdb_text: str, selected_chain: str = None):
    try:
        from Bio.PDB import PDBParser
    except Exception as e:
        raise RuntimeError("Biopython is required for PDB parsing.") from e

    from io import StringIO
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("uploaded", StringIO(pdb_text))

    coords = []
    resnos = []
    chain_ids = []

    for model in structure:
        for chain in model:
            if selected_chain and chain.id != selected_chain:
                continue
            for res in chain:
                if "CA" in res:
                    coords.append(res["CA"].get_coord().astype(np.float32))
                    resnos.append(res.id[1])
                    chain_ids.append(chain.id)
            if selected_chain:
                break
        break

    if len(coords) == 0:
        raise ValueError("No Cα atoms found for the selected chain.")

    return np.array(coords, dtype=np.float32), resnos, chain_ids


def contact_map(coords: np.ndarray, size: int = 128):
    if coords.size == 0:
        raise ValueError("No coordinates available for contact map.")
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    mat = 1.0 / (1.0 + dist)
    out = np.zeros((size, size), dtype=np.float32)
    s = min(mat.shape[0], size)
    out[:s, :s] = mat[:s, :s]
    return out


def residue_index_from_position(resnos, pos: int):
    if pos in resnos:
        return resnos.index(pos)
    return min(max(pos - 1, 0), len(resnos) - 1)


def build_cnn_input_from_pdb(pdb_text: str, pos: int, chain_id: str = None):
    coords, resnos, _ = pdb_to_ca_coords(pdb_text, selected_chain=chain_id)
    cm = contact_map(coords, CNN_INPUT_SIZE)
    idx = residue_index_from_position(resnos, pos)

    # Mutation-aware marking: row and column at mutated residue are highlighted.
    x = cm.copy()
    if idx < x.shape[0]:
        x[idx, :] = 1.0
        x[:, idx] = 1.0

    return x[None, ..., None].astype(np.float32)


def build_gnn_inputs_from_pdb(pdb_text: str, pos: int, chain_id: str = None):
    coords, resnos, _ = pdb_to_ca_coords(pdb_text, selected_chain=chain_id)
    idx = residue_index_from_position(resnos, pos)

    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    adj = (dist < 8.0).astype(np.float32)
    np.fill_diagonal(adj, 1.0)

    deg = adj.sum(axis=1)
    deg_inv = np.diag(1.0 / np.maximum(deg, 1e-6))
    a_hat = (deg_inv @ adj).astype(np.float32)

    n = adj.shape[0]
    x = np.zeros((n, 2), dtype=np.float32)
    x[:, 0] = deg / max(float(deg.max()), 1.0)
    x[idx, 1] = 1.0

    return a_hat, x, idx


# -----------------------------
# Prediction functions
# -----------------------------
def predict_xgb(rows):
    booster, err = load_xgb_model()
    if booster is None:
        return [None] * len(rows), err

    try:
        xdf = build_xgb_features(pd.DataFrame(rows))
        dmat = xgb.DMatrix(xdf, feature_names=list(xdf.columns))
        return [float(x) for x in booster.predict(dmat)], None
    except Exception as e:
        return [None] * len(rows), f"XGB prediction failed: {e}"


def predict_cnn_for_row(pdb_text, row, chain_id=None):
    model, err = load_cnn_model()
    if model is None:
        return None, err

    try:
        x = build_cnn_input_from_pdb(pdb_text, int(row["position"]), chain_id=chain_id)
        y = model.predict(x, verbose=0).ravel()[0]
        return float(y), None
    except Exception as e:
        return None, f"CNN prediction failed for {row.get('mutation', '')}: {e}"


def predict_gnn_for_row(pdb_text, row, chain_id=None):
    loaded, err = load_gnn_model()
    if loaded is None:
        return None, err

    try:
        import torch
        gnn, device = loaded
        a_hat, x, idx = build_gnn_inputs_from_pdb(pdb_text, int(row["position"]), chain_id=chain_id)
        a_t = torch.from_numpy(a_hat).float().to(device)
        x_t = torch.from_numpy(x).float().to(device)
        idx_t = torch.tensor([idx], dtype=torch.int64).to(device)

        with torch.no_grad():
            try:
                out = gnn(a_t, x_t, idx_t)
            except Exception:
                out = gnn(a_t, x_t)

        return float(out.detach().cpu().numpy().ravel()[0]), None
    except Exception as e:
        return None, f"GNN prediction failed for {row.get('mutation', '')}: {e}"


def weighted_ensemble(preds: dict):
    """
    Weighted ensemble with re-normalization over available valid model outputs.
    If only XGB is available, result = XGB.
    If XGB + CNN + GNN are available, result = 0.45*XGB + 0.35*CNN + 0.20*GNN.
    If one model fails, remaining weights are re-normalized.
    """
    num = 0.0
    den = 0.0

    for model_name, pred in preds.items():
        if pred is not None and np.isfinite(pred):
            w = ENSEMBLE_WEIGHTS.get(model_name, 0.0)
            num += w * float(pred)
            den += w

    if den == 0:
        return None
    return float(num / den)


def interpret_ddg(ddg):
    if ddg is None or not np.isfinite(ddg):
        return "Not available"
    if ddg < -0.5:
        return "Potentially stabilizing"
    if ddg > 0.5:
        return "Potentially destabilizing"
    return "Near-neutral / weak effect"


def build_rows_for_scan(seq: str, start_pos: int, end_pos: int):
    rows = []
    for pos in range(start_pos, end_pos + 1):
        if pos < 1 or pos > len(seq):
            continue
        wt = seq[pos - 1]
        if wt not in AA:
            continue
        for mt in AA:
            if mt == wt:
                continue
            rows.append({
                "protein_id": "USER",
                "wt": wt,
                "position": pos,
                "mt": mt,
                "mutation": f"{wt}{pos}{mt}"
            })
    return rows


def score_rows(rows, pdb_text=None, chain_id=None, progress_bar=None):
    xgb_vals, xgb_err = predict_xgb(rows)
    use_structure = bool(pdb_text)

    results = []
    errors = []
    if xgb_err:
        errors.append(xgb_err)

    total = len(rows)
    for i, row in enumerate(rows):
        if progress_bar is not None and total:
            progress_bar.progress((i + 1) / total)

        preds = {
            "xgb": xgb_vals[i] if i < len(xgb_vals) else None,
            "cnn": None,
            "gnn": None
        }

        if use_structure:
            cnn_pred, cnn_err = predict_cnn_for_row(pdb_text, row, chain_id=chain_id)
            gnn_pred, gnn_err = predict_gnn_for_row(pdb_text, row, chain_id=chain_id)
            preds["cnn"] = cnn_pred
            preds["gnn"] = gnn_pred

            # Avoid repeated error spam; keep first few unique messages.
            for err in [cnn_err, gnn_err]:
                if err and err not in errors and len(errors) < 5:
                    errors.append(err)

        ens = weighted_ensemble(preds)

        valid_vals = [v for v in preds.values() if v is not None and np.isfinite(v)]
        disagreement = float(np.std(valid_vals)) if len(valid_vals) > 1 else 0.0

        results.append({
            "mutation": row["mutation"],
            "position": row["position"],
            "wt": row["wt"],
            "mt": row["mt"],
            "XGB_ddG": preds["xgb"],
            "CNN_ddG": preds["cnn"],
            "GNN_ddG": preds["gnn"],
            "Ensemble_ddG": ens,
            "ensemble_std": disagreement,
            "interpretation": interpret_ddg(ens)
        })

    return pd.DataFrame(results), errors


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("ΔΔG Mutation Scanner")
st.caption("Phase 1 weighted ensemble: XGB 0.45, CNN 0.35, GNN 0.20")

with st.expander("Model status", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.write("XGB model")
    col1.success("Found" if os.path.exists(XGB_PATH) else "Missing")
    col2.write("CNN model")
    col2.success("Found" if os.path.exists(CNN_PATH) else "Missing")
    col3.write("GNN model")
    col3.success("Found" if os.path.exists(GNN_PATH) else "Missing")

    st.write("Ensemble weights:", ENSEMBLE_WEIGHTS)
    st.caption("If only FASTA is supplied, the tool uses XGB. If PDB is supplied and CNN/GNN load successfully, the weighted ensemble is calculated. Missing model outputs are automatically excluded and remaining weights are re-normalized.")

st.sidebar.header("Input")

fasta_file = st.sidebar.file_uploader("Upload FASTA file", type=["fasta", "fa", "txt"])
fasta_text = st.sidebar.text_area("Or paste FASTA / protein sequence", height=160)

pdb_file = st.sidebar.file_uploader("Optional: upload PDB file for CNN/GNN structure-aware prediction", type=["pdb"])
chain_id = st.sidebar.text_input("PDB chain ID (optional)", value="")

pdb_text = ""
if pdb_file is not None:
    pdb_text = pdb_file.read().decode("utf-8", errors="ignore")

if fasta_file is not None:
    fasta_input = fasta_file.read().decode("utf-8", errors="ignore")
else:
    fasta_input = fasta_text

seq = parse_fasta(fasta_input)

if seq:
    st.sidebar.success(f"Sequence loaded: {len(seq)} amino acids")
else:
    st.sidebar.warning("Please upload or paste a FASTA/protein sequence.")

if pdb_text:
    st.sidebar.success("PDB file loaded")
else:
    st.sidebar.info("PDB not supplied. FASTA-only mode will use XGB.")

chain_id_clean = chain_id.strip() or None

tab1, tab2 = st.tabs(["Single mutation prediction", "19-amino-acid scan"])

with tab1:
    st.subheader("Single mutation prediction")

    if not seq:
        st.info("Upload or paste a FASTA/protein sequence to continue.")
    else:
        mutations = st.text_input("Mutation(s)", value="", placeholder="Example: I38D or I38D;A42V")

        if st.button("Predict mutation(s)", type="primary"):
            rows = parse_mutations(mutations)
            rows = [r for r in rows if 1 <= r["position"] <= len(seq)]

            # Warn if WT does not match FASTA, but do not stop.
            for r in rows:
                seq_wt = seq[r["position"] - 1]
                if seq_wt != r["wt"]:
                    st.warning(f"Mutation {r['mutation']}: WT residue in FASTA is {seq_wt}, but mutation input says {r['wt']}.")

            if not rows:
                st.error("No valid mutation found. Please enter mutations like I38D or I38D;A42V.")
            else:
                with st.spinner("Running prediction..."):
                    df_result, errors = score_rows(rows, pdb_text=pdb_text, chain_id=chain_id_clean)

                st.markdown("### Prediction results")
                st.dataframe(df_result, use_container_width=True)

                if errors:
                    with st.expander("Warnings / model messages"):
                        for err in errors:
                            st.write(err)

                csv = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download prediction CSV",
                    data=csv,
                    file_name="single_mutation_predictions.csv",
                    mime="text/csv"
                )

with tab2:
    st.subheader("19-amino-acid scan")

    if not seq:
        st.info("Upload or paste a FASTA/protein sequence to continue.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            start_pos = st.number_input("Start position", min_value=1, max_value=len(seq), value=1, step=1)
        with c2:
            end_default = min(len(seq), int(start_pos) + 19)
            end_pos = st.number_input("End position", min_value=1, max_value=len(seq), value=end_default, step=1)

        if st.button("Run 19-AA scan", type="primary"):
            start_pos = int(start_pos)
            end_pos = int(end_pos)

            if end_pos < start_pos:
                st.error("End position must be greater than or equal to start position.")
            else:
                rows = build_rows_for_scan(seq, start_pos, end_pos)
                st.write(f"Mutations to score: {len(rows)}")

                progress = st.progress(0.0)
                with st.spinner("Running 19-AA scan..."):
                    df_scan, errors = score_rows(rows, pdb_text=pdb_text, chain_id=chain_id_clean, progress_bar=progress)

                df_ranked = df_scan.sort_values("Ensemble_ddG", ascending=True).reset_index(drop=True)
                df_ranked.insert(0, "Rank", np.arange(1, len(df_ranked) + 1))

                st.markdown("### Full scan results ranked by ensemble ΔΔG")
                st.dataframe(df_ranked, use_container_width=True)

                st.markdown("### Top 10 stabilizing candidates")
                top10 = df_ranked[df_ranked["Ensemble_ddG"].notna()].head(10)
                st.dataframe(top10, use_container_width=True)

                if errors:
                    with st.expander("Warnings / model messages"):
                        for err in errors:
                            st.write(err)

                csv_full = df_ranked.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download full scan CSV",
                    data=csv_full,
                    file_name="weighted_ensemble_scan_results.csv",
                    mime="text/csv"
                )

                csv_top = top10.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download top-10 CSV",
                    data=csv_top,
                    file_name="top10_weighted_ensemble_candidates.csv",
                    mime="text/csv"
                )

                if len(df_ranked) > 0:
                    st.markdown("### Ensemble ΔΔG plot")
                    plot_df = df_ranked[["mutation", "Ensemble_ddG"]].dropna().set_index("mutation")
                    st.bar_chart(plot_df)
