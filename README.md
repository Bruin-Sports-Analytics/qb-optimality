# QB Clutch Optimality

**Research question:** Which quarterbacks stay closest to optimal pocket behavior when the game is on the line, and which ones break down?

## Overview

We train an XGBoost model on spatial features derived from NFL player tracking data to predict EPA (Expected Points Added) per play. The model learns what pocket behaviors historically produce good outcomes. For each play we compute an **Optimality Score** = Actual EPA − Predicted EPA, then compare each QB's score in clutch vs. non-clutch situations to produce a **Clutch Optimality Rating**.

Inspired by Brian Burke's DeepQB (ESPN / MIT Sloan 2019), extended to pocket behavior and clutch situational context.

---

## Clutch Definition

Set in `00_download_pbp.py`:

| Label | Rule |
|-------|------|
| Clutch | Q4 or OT **and** any of: win probability 40–60%; 4th down with \|score diff\| ≤ 8; trailing by ≤ 8 with ≤ 120 s left |
| Non-clutch | Win probability < 20% or > 80% (and not clutch) |
| Neutral | Everything else |

**Clutch Optimality Rating** = mean(Optimality Score in clutch) − mean(Optimality Score across all of the QB's plays)  
Positive → QB outperforms model expectations more when the game is close than he does on average.

---

## Data Sources

| Source | Contents | Location |
|--------|----------|----------|
| nfl_data_py (nflfastR) | 2021 play-by-play, EPA, win probability | `data/raw/pbp_2021.parquet` |
| NFL Big Data Bowl 2023 (Kaggle) | 10 Hz player tracking, weeks 1–8 of 2021 | `data/raw/big_data_bowl_2023/` |

Big Data Bowl files: `games.csv`, `plays.csv`, `players.csv`, `pffScoutingData.csv`, `tracking_week_1.csv` … `tracking_week_8.csv`

> **Note:** `data/raw/` and `data/processed/` are git-ignored. See setup steps below.

---

## Directory Layout

```
qb-optimality/
├── data/
│   ├── raw/
│   │   ├── pbp_2021.parquet          # nflfastR play-by-play
│   │   └── big_data_bowl_2023/       # Kaggle tracking data
│   └── processed/                    # Merged, feature-engineered data
├── outputs/
│   ├── figures/                      # Plots and charts
│   └── tables/                       # CSV summaries and rankings
├── notebooks/                        # Exploratory analysis
├── 00_download_pbp.py                # Download nflfastR PBP data
├── 01_validate_data.py               # Validate all data files are present
├── features/                         # qb_features.py, rusher_features.py, pocket_features.py
├── 02_feature_engineering.py         # Join tracking to PBP, extract snap/release frames, build features
├── 03_model_training.py              # Train XGBoost EPA model (leave-one-week-out CV)
├── 04_analysis.py                    # Optimality scores + QB clutch ratings
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download nflfastR play-by-play data
```bash
python 00_download_pbp.py
```
Saves filtered 2021 QB dropback data to `data/raw/pbp_2021.parquet`.

### 3. Download Big Data Bowl 2023 tracking data (Kaggle)

**One-time Kaggle credentials setup:**
1. Go to kaggle.com → Account → "Create New Token" → copies a `KGAT_...` token to your clipboard
2. Save it: `mkdir -p ~/.kaggle && echo "PASTE_YOUR_TOKEN_HERE" > ~/.kaggle/access_token`
3. Lock permissions: `chmod 600 ~/.kaggle/access_token`

**Download and unzip:**
```bash
kaggle competitions download -c nfl-big-data-bowl-2023 -p data/raw/big_data_bowl_2023/
unzip data/raw/big_data_bowl_2023/nfl-big-data-bowl-2023.zip -d data/raw/big_data_bowl_2023/
```
> You must accept the competition rules on Kaggle before the download will work.

### 4. Validate all data
```bash
python 01_validate_data.py
```
All 13 file checks should pass before proceeding.

---

## Engineered Features (at moment of pass release)

| Category | Features |
|----------|---------|
| QB | Displacement from snap, speed at release, orientation at release, time from snap to release |
| Pass rushers | Distance of nearest rusher, approach speed of nearest 2 rushers, rushers within 3 yards |
| Pocket | Pocket area (convex hull of OL), pocket collapse rate (Δarea snap→release) |
| Context | Down, distance, yardline, score differential, half_seconds_remaining |

---

## Model

- **Algorithm:** XGBoost regressor
- **Target:** EPA per play
- **Validation:** leave-one-week-out across weeks 1–8 (the model never scores a week it trained on)
- **Sample size:** 7,088 QB dropbacks (weeks 1–8, 2021)
- **Tuned hyperparameters:** 200 trees, max depth 3, learning rate 0.05

Neural networks were ruled out — sample size is too small for reliable generalization at this scope.

---

## Pipeline (in order)

```
00_download_pbp.py          →  data/raw/pbp_2021.parquet
kaggle download + unzip     →  data/raw/big_data_bowl_2023/
01_validate_data.py         →  confirms all inputs present
02_feature_engineering.py   →  data/processed/features.parquet
03_model_training.py        →  outputs/model.json, outputs/tables/model_summary.json, play_predictions.csv
04_analysis.py              →  outputs/tables/qb_clutch_ratings.csv
```

---

## Results

**Model:** leave-one-week-out RMSE of **1.57 EPA** overall (1.60 on the held-out test week), with
per-week RMSE between 1.50 and 1.70. EPA per play is noisy, so the model's job is to set a fair
expectation for each dropback, not to predict it exactly.

**Clutch ratings** (QBs with ≥10 clutch and ≥50 total dropbacks; full table in
`outputs/tables/qb_clutch_ratings.csv`):

| QB | Clutch rating | Clutch dropbacks |
|---|---:|---:|
| J. Herbert | +0.53 | 19 |
| J. Goff | +0.43 | 11 |
| L. Jackson | +0.40 | 14 |
| … | | |
| K. Cousins | −0.30 | 41 |
| T. Brady | −0.44 | 20 |
| C. Wentz | −0.50 | 19 |

**Caveat:** clutch samples are small (11–41 dropbacks per QB, from 8 weeks of tracking data), so these
are a first look, not a ranking. Next steps are bootstrapped intervals and more seasons of tracking data.

Write-up: https://eshaandhavala.github.io/entries/qb-clutch-optimality/

---

## Team

Bruin Sports Analytics football research, spring 2026.

- **Eshaan Dhavala:** integration and feature-engineering pipeline, pass-rusher features, leakage and bug fixes, final model and ratings
- **Abhi Kumar:** pocket geometry features
- **Andrew:** model training script
- **Keith Bui:** optimality and clutch analysis
- **Dillon Maheshwari:** QB spatial features
- **Gonzalo Merino Sanchez:** first draft of pass-rusher features
