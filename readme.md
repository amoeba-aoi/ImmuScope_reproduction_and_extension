# Dong Yizhe Capstone Code Notes (English)

> Note: All commands below are assumed to run from the project root (the directory containing `configs/`, `ImmuScope/`, and `scripts/`).

### A. Utility Scripts (Non-`main_*`)

#### `auc_from_cd4_per_sample.py`

Purpose:
- Compute three CD4 metrics from `results_pred_per_sample.csv`:
  - protein-level Median AUC
  - protein-level Mean AUC
  - Overall AUC
- Outputs:
  - `results_pred_protein_avg.csv`
  - `results_auc_protein_avg.csv`

Example:
```bash
python auc_from_cd4_per_sample.py --csv results/NetMHCIIpan-CD4/results_pred_per_sample.csv
```

---

#### `build_context12_from_h5.py`

Purpose:
- Build MixMHC2pred-required `context12` (6 aa left + 6 aa right) from H5 + FASTA
- Can export:
  - Full CSV (including `mhc_names, peptide, context12, label, protein_id`, etc.)
  - Two-column text: `peptide context12`

Example:
```bash
python build_context12_from_h5.py \
  --h5 data/train_test_h5py/NetMHCIIpan_eval.h5 \
  --fasta data/raw/NetMHCIIpan_eval.fa \
  --out-csv data/tmp_datasets/mixmhc2pred_input_with_context12.csv \
  --out-two-col data/tmp_datasets/mixmhc2pred_input_two_col.txt
```

---

#### `compare_cd4_auc_three.py`

Purpose:
- Compare CD4 AUC across ImmuScope / NetMHCIIpan / MixMHC2pred
- Supports two modes:
  - `intersection` (default): compare after inner join using aligned keys (recommended)
  - `full`: compute each method independently

Example:
```bash
python compare_cd4_auc_three.py --mode intersection
python compare_cd4_auc_three.py --mode full
```

---

#### `export_immuscope_cd4_per_sample.py`

Purpose:
- Reuse the CD4 test flow to export ImmuScope per-sample predictions:
  - `results/ImmuScope-CD4/results_pred_per_sample.csv`
- Includes CPU-safe checkpoint loading (`map_location=cpu` when no GPU) and progress logging

Example:
```bash
python export_immuscope_cd4_per_sample.py \
  -d configs/data.yaml -m configs/ImmuScope.yaml -s 0 -n 10
```

---

#### `iedb_tcell_export_to_h5.py`

Purpose:
- Clean IEDB T-cell assay exports (`.xlsx`/`.csv`) and optionally convert to ImmuScope `.h5`
- Supports train/test split and context-group-based split

Example:
```bash
python iedb_tcell_export_to_h5.py \
  -i data/raw/iedb/tcell_table_export.xlsx --header-rows 2 \
  --mhc-seq data/raw/pseudosequence.2023.dat \
  -o data/iedb_clean/tcell_clean.tsv \
  --h5 data/iedb_h5/tcell_eval.h5
```

Dependency:
- `openpyxl` is required for reading `.xlsx`

---

#### `inferential_stats_sections.py`

Purpose:
- Generate inferential statistics for Sections 3.1–3.6 (bootstrap / Friedman / Wilcoxon / Holm, etc.)
- Adds paired AUPR and fixed-threshold PPV analyses
- Output directory: `results/inferential_stats/`

Example:
```bash
python inferential_stats_sections.py \
  --reproduction-dir ... --ablation-dir ... --project-root ...
```

---

#### `merge_imm_train_h5.py`

Purpose:
- Merge official IM training data with additional IEDB training data (optional deduplication)
- Dedup key: `(mhc, peptide, context)`; official rows are kept by default when duplicated

Example:
```bash
python merge_imm_train_h5.py \
  --official data/im_datasets/imm_train.h5 \
  --extra data/iedb_clean/ag85a_tcell_train.h5 \
  -o data/im_datasets/imm_train_merged_official_iedb.h5
```

---

#### `motif_deconv_compare.py`

Purpose:
- Perform motif deconvolution and cross-method comparison for ImmuScope / NetMHCIIpan / MixMHC2pred
- Input: three CSVs containing `mhc_names, peptide, pred`
- Output: PWM, logo, pairwise JSD/KL, and summary tables

Example:
```bash
python motif_deconv_compare.py \
  --immuscope-csv results/ImmuScope-CD4/results_pred_per_sample.csv \
  --netmhc-csv results/NetMHCIIpan-CD4/results_pred_per_sample.csv \
  --mixmhc-csv results/MixMHC2pred-CD4/results_pred_per_sample.csv \
  --out-dir results/motif_compare
```

---

#### `run_downstream_cascade_ablation.py`

Purpose:
- Batch-launch downstream CD4/IM training from EL variants
- EL variants are configured in `VARIANTS` with `weights_tag / el_stem / el_suffix`

Example:
```bash
python run_downstream_cascade_ablation.py --cd4-only --start-id 0 --num-models 1
python run_downstream_cascade_ablation.py --im-only
```

---

#### `run_netmhc_cd4_baseline.py`

Purpose:
- Run NetMHCIIpan baseline on CD4 benchmark H5
- Supports Docker backend (default) and local binary backend
- Outputs result tables aligned with the CD4 test protocol

Example (Docker):
```bash
python run_netmhc_cd4_baseline.py \
  --data-cnf configs/data.yaml \
  --docker-image ghcr.io/macromnex/netmhc2pan_mcp:latest \
  --platform linux/amd64
```

Example (local binary):
```bash
python run_netmhc_cd4_baseline.py --backend local --netmhc-bin /path/to/netMHCIIpan
```

---

### B. Notebook Descriptions

#### `ImmuScope_ablation.ipynb`
- Goal: full ablation workflow (EL variant training, evaluation, and export)
- Environment: Colab-style, typically includes Drive mount, repo clone, dependency install, and main script execution

#### `ImmuScope_ablation_A1.ipynb`
- Goal: A1 sub-experiment / single ablation track reproduction
- Environment: similar to above, focused on quick verification of one configuration chain

#### `immuscope_reproduction.ipynb`
- Goal: one-click reproduction of the original pipeline (EL/CD4/IM)

#### `immuscope_reproduction_EL_CD4_immunogenicity.ipynb`
- Goal: explicit end-to-end reproduction of EL -> CD4 -> IM

#### `immuscope_reproduction_different_division.ipynb`
- Goal: analyze impact of different data partitioning strategies (split variants)

#### `immuscope_immunogenicity_new_peptide_data.ipynb`
- Goal: immunogenicity workflow validation/testing on new peptide data

Notebook usage recommendations:
- First verify path mapping (Drive/local paths) is consistent with `configs/data.yaml`
- If running locally, replace Colab-specific commands (e.g., `drive.mount`) with local paths

---

### C. Modified Main Entrypoints: `main_*_dongyizhe_20260405.py`

- `main_antigen_presentation_5cv_dongyizhe_20260405.py`
- `main_antigen_presentation_train_dongyizhe_20260405.py`
- `main_cd4_epitope_test_dongyizhe_20260405.py`
- `main_cd4_epitope_train_dongyizhe_20260405.py`
- `main_immunogenicity_test_dongyizhe_20260405.py`
- `main_immunogenicity_train_dongyizhe_20260405.py`

Overall purpose of these entrypoints:
- Extended from original main scripts with EL variant weight selection, eval-only mode, skip-pretrain options, and tagged outputs for batch ablation/cascade experiments.

---

## 3) Recommended Execution Order

1. Prepare/verify paths in `configs/data.yaml`  
2. Run EL (or directly use existing EL checkpoints)  
3. Export/generate CD4 per-sample outputs (`export_immuscope_cd4_per_sample.py`)  
4. Run baseline and three-method comparison (`run_netmhc_cd4_baseline.py`, `compare_cd4_auc_three.py`)  
5. Run motif and inferential statistics (`motif_deconv_compare.py`, `inferential_stats_sections.py`)  
6. For additional IEDB training: run `iedb_tcell_export_to_h5.py` + `merge_imm_train_h5.py`, then run IM

