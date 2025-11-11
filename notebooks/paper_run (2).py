#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install kagglehub')
get_ipython().system('pip install dotenv')


# ### Get the Dataset

# In[ ]:


import os
import kagglehub

path = kagglehub.dataset_download("chiranjivdas09/ta-feng-grocery-dataset")
print("Dataset path:", path)

with open(".env", "w") as f:
    f.write(f"DATASET_PATH={path}\n")


# ### Verify

# In[ ]:


from dotenv import load_dotenv
import os

load_dotenv()
path = os.getenv("DATASET_PATH")

if not path:
    print("FAILED TO DOWNLOAD DATASET — DATASET_PATH not found or empty.")
else:
    print("Dataset path loaded successfully:")
    print(path)


# ### 📊 EDA — What this section does
# 
# **Goal.** Understand the Ta-Feng dataset’s shape, data quality, and basic patterns to choose sensible preprocessing and mining settings.
# 
# **Inputs.**
# - `ta_feng_all_months_merged.csv` (9 columns): `TRANSACTION_DT, CUSTOMER_ID, AGE_GROUP, PIN_CODE, PRODUCT_SUBCLASS, PRODUCT_ID, AMOUNT, ASSET, SALES_PRICE`
# 
# **What happens (step-by-step).**
# 1. **Load & standardize columns** — Uppercase names; coerce types (IDs → strings; `AMOUNT/ASSET/SALES_PRICE` → numeric; `TRANSACTION_DT` → datetime).
# 2. **Coverage checks** — Print min/max date, null counts, and basic stats for numeric columns.
# 3. **Temporal features for inspection** — Derive `YEAR_MONTH`, `WEEK`, `DAY_OF_WEEK` (for slicing/plots only).
# 4. **Top-K distributions** — Frequency tables for `PRODUCT_ID`, `PRODUCT_SUBCLASS`, `AGE_GROUP`, `PIN_CODE`, `YEAR_MONTH`, `DAY_OF_WEEK`.
# 5. **Skew-aware histograms** — For `AMOUNT`, `ASSET`, `SALES_PRICE`, show **Full** vs **Clipped @ 99th percentile** histograms side-by-side to make long tails interpretable.
# 6. **Basket construction** — Group lines into **transactions** by (`CUSTOMER_ID`, `TRANSACTION_DT`); collect unique item lists per transaction (`items`) and compute `basket_size`.
# 7. **Co-occurrence peek** — Top item pairs by co-appearance (sanity check before true itemset mining).
# 8. **Temporal volume** — Bar plots for transactions per week/month (seasonality/holidays).
# 9. **Cache artifacts**  
#    - `artifacts/eda/transactions_normalized.parquet` — clean, typed transactions  
#    - `artifacts/eda/baskets.parquet` — per-transaction `items` + `basket_size`
# 
# **Why this matters.**
# - Confirms time horizon & sparsity.
# - Reveals heavy-tail behavior (justifies winsorization later).
# - Validates basket construction (essential for frequent itemset mining).
# 

# In[ ]:


# === EDA for Ta-Feng (exact 9-column schema) ===
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 4)

DATASET_DIR = Path(os.getenv("DATASET_PATH") or ".").resolve()
CSV_PATH = None
# Prefer the merged file; fall back to first CSV if path varies
candidates = list(DATASET_DIR.rglob("ta_feng_all_months_merged.csv"))
if candidates:
    CSV_PATH = candidates[0]
else:
    all_csvs = list(DATASET_DIR.rglob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSVs found beneath: {DATASET_DIR}")
    # heuristically pick the largest csv (often the merged)
    CSV_PATH = max(all_csvs, key=lambda p: p.stat().st_size)

print("Using CSV:", CSV_PATH)

# 1) Load & normalize column names to snake_case
df = pd.read_csv(CSV_PATH, low_memory=False)
orig_cols = list(df.columns)
df.columns = [c.strip().upper() for c in df.columns]

expected = [
    "TRANSACTION_DT","CUSTOMER_ID","AGE_GROUP","PIN_CODE",
    "PRODUCT_SUBCLASS","PRODUCT_ID","AMOUNT","ASSET","SALES_PRICE"
]
missing_expected = [c for c in expected if c not in df.columns]
if missing_expected:
    print("WARNING: missing expected columns:", missing_expected)

# Keep only the 9 we care about (if extras exist)
cols = [c for c in expected if c in df.columns]
df = df[cols].copy()

# 2) Dtypes: IDs as strings; numeric as numeric; parse datetime
# TRANSACTION_DT can be like '11/01/2000' or '1Nov00'—use coerce
def parse_dt(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)

df["TRANSACTION_DT"] = parse_dt(df["TRANSACTION_DT"])

for c in ["CUSTOMER_ID","PRODUCT_SUBCLASS","PRODUCT_ID","PIN_CODE","AGE_GROUP"]:
    if c in df.columns:
        df[c] = df[c].astype(str)

for c in ["AMOUNT","ASSET","SALES_PRICE"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

print("\n=== Schema preview ===")
print(df.head(3))
print("\nInfo:")
print(df.info())

print("\nMissing values (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))

num_cols = [c for c in ["AMOUNT","ASSET","SALES_PRICE"] if c in df.columns]
if num_cols:
    print("\nBasic numeric stats:")
    display(df[num_cols].describe(percentiles=[.25,.5,.75,.9,.95,.99]).T)

# 3) Temporal enrichments
df["YEAR_MONTH"] = df["TRANSACTION_DT"].dt.to_period("M").astype(str)
df["WEEK"] = df["TRANSACTION_DT"].dt.to_period("W").astype(str)
df["DAY_OF_WEEK"] = df["TRANSACTION_DT"].dt.day_name()

print("\nTemporal coverage:")
print("min date:", df["TRANSACTION_DT"].min())
print("max date:", df["TRANSACTION_DT"].max())

# 4) Quick distributions / counts
def top_counts(col, k=10):
    if col in df.columns:
        vc = df[col].value_counts().head(k)
        print(f"\nTop {k} {col}:")
        display(vc)

top_counts("PRODUCT_ID")
top_counts("PRODUCT_SUBCLASS")
top_counts("AGE_GROUP")
top_counts("PIN_CODE")
top_counts("YEAR_MONTH")
top_counts("DAY_OF_WEEK")

# 5) Histograms for numeric fields: NORMAL vs CLIPPED side-by-side
def plot_normal_and_clipped(series, title, bins=50, upper_q=0.99):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        print(f"Skipping {title}: no data.")
        return
    q_hi = s.quantile(upper_q)
    s_clip = s.clip(upper=q_hi)
    n_clipped = int((s > q_hi).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(s, bins=bins)
    axes[0].set_title(f"{title} — Full")
    axes[0].set_xlabel(title); axes[0].set_ylabel("freq")

    axes[1].hist(s_clip, bins=bins)
    axes[1].set_title(f"{title} — Clipped ≤ {upper_q:.0%} (>{n_clipped} clipped)")
    axes[1].set_xlabel(title); axes[1].set_ylabel("freq")
    plt.tight_layout()
    plt.show()

for c in num_cols:
    plot_normal_and_clipped(df[c], c, bins=50, upper_q=0.99)

# 6) Basket construction:
# Use (CUSTOMER_ID, TRANSACTION_DT) as transaction key.
# If times are date-only (no hours), this still groups per customer-day (or per customer-timestamp if present).
key_cols = ["CUSTOMER_ID", "TRANSACTION_DT"] if "CUSTOMER_ID" in df.columns else ["TRANSACTION_DT"]

# Choose granularity: product_id (fine) vs product_subclass (coarser)
# Force subclass-level items (coarser, more interpretable)
# ITEM_COL = "PRODUCT_SUBCLASS"
ITEM_COL = "PRODUCT_ID" if "PRODUCT_ID" in df.columns else "PRODUCT_SUBCLASS"
baskets = (df.dropna(subset=[ITEM_COL])
             .groupby(key_cols)[ITEM_COL]
             .apply(lambda s: list(pd.unique(s.astype(str))))
             .reset_index(name="items"))

print("\nBaskets preview:")
print(baskets.head(5))
print("Num baskets:", len(baskets))

# Basket size
baskets["basket_size"] = baskets["items"].apply(len)
baskets["basket_size"].plot(kind="hist", bins=30, title="Basket size distribution")
plt.xlabel("distinct items per transaction"); plt.ylabel("freq"); plt.show()

print("\nBasket size stats:")
display(baskets["basket_size"].describe(percentiles=[.5,.75,.9,.95,.99]))

# 7) Sanity checks: duplicate transactions, extreme outliers
dups = baskets.duplicated(subset=key_cols).sum()
print("\nPotential duplicate transaction keys:", int(dups))

for c in num_cols:
    q99 = df[c].quantile(0.99)
    mx = df[c].max()
    print(f"{c}: 99th pct={q99:.2f}, max={mx:.2f}")

# 8) Quick co-occurrence peek (cap for speed)
from collections import Counter
from itertools import combinations
pair_counts = Counter()
for items in baskets["items"].tolist()[:200000]:  # cap to keep it quick
    s = sorted(set(items))
    for a, b in combinations(s, 2):
        pair_counts[(a,b)] += 1

top = pd.DataFrame([(a,b,c) for (a,b),c in pair_counts.most_common(20)],
                   columns=[f"{ITEM_COL}_1", f"{ITEM_COL}_2", "pair_count"])
print("\nTop co-occurring pairs:")
display(top)

# 9) Weekly and monthly basket counts
wk = baskets.copy()
wk["WEEK"] = wk["TRANSACTION_DT"].dt.to_period("W").astype(str)
wk_counts = wk["WEEK"].value_counts().sort_index()
wk_counts.plot(kind="bar", title="Transactions per week"); plt.xlabel("week"); plt.ylabel("#transactions"); plt.show()

mo_counts = baskets["TRANSACTION_DT"].dt.to_period("M").astype(str).value_counts().sort_index()
mo_counts.plot(kind="bar", title="Transactions per month"); plt.xlabel("year-month"); plt.ylabel("#transactions"); plt.show()

# 10) Cache artifacts for downstream steps
OUT_DIR = Path("artifacts/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tidy normalized transactions (snake_case, standard dtypes)
df_out = df.rename(columns={
    "TRANSACTION_DT":"transaction_dt",
    "CUSTOMER_ID":"customer_id",
    "AGE_GROUP":"age_group",
    "PIN_CODE":"pin_code",
    "PRODUCT_SUBCLASS":"product_subclass",
    "PRODUCT_ID":"product_id",
    "AMOUNT":"amount",
    "ASSET":"asset",
    "SALES_PRICE":"sales_price",
    "YEAR_MONTH":"year_month",
    "WEEK":"week",
    "DAY_OF_WEEK":"day_of_week",
})

df_out_path = OUT_DIR / "transactions_normalized.parquet"
baskets_path = OUT_DIR / "baskets.parquet"
df_out.to_parquet(df_out_path, index=False)
baskets.to_parquet(baskets_path, index=False)

print(f"\nSaved normalized transactions → {df_out_path}")
print(f"Saved baskets → {baskets_path}")

print("""
Report talking points to fill (after inspecting outputs):
• Temporal: confirm coverage (2000-11-01 to 2001-02-28), weekly peaks (e.g., holidays), and month-to-month shifts.
• Product mix: head vs long tail (PRODUCT_ID / PRODUCT_SUBCLASS), any notable families.
• Demographics: AGE_GROUP and PIN_CODE distributions; % missing in AGE_GROUP (~3%).
• Sales metrics: AMOUNT/ASSET/SALES_PRICE distributions; identify extreme outliers (list 99th pct vs max).
• Basket structure: typical/median size; tail behavior (95th/99th percentiles); implications for itemset mining.
""")


# ### 🧹 Data Preprocessing — What this section does
# 
# **Goal.** Turn raw transactions into **clean, model-ready features** for clustering/dimensionality reduction and downstream mining, while keeping a minimally imputed facts table.
# 
# **Inputs.**
# - `artifacts/eda/transactions_normalized.parquet`  
# - `artifacts/eda/baskets.parquet`
# 
# **What happens (step-by-step).**
# 1. **Missing-data handling**
#    - Drop rows missing transaction keys (`transaction_dt`, and `customer_id` if present).
#    - Impute categoricals: `AGE_GROUP`/`PIN_CODE` → `"Unknown"`.
#    - Keep numeric NaNs for now; median-impute later inside pipelines.
# 2. **Feature engineering (transaction-level)**
#    - Aggregate per transaction: `total_amount`, `total_asset`, `total_sales`, `n_lines` (row count), `basket_size` (distinct items), `avg_unit_price = total_sales / total_amount`.
#    - Attach lightweight context: `year_month`, `day_of_week`, `week`, plus demographics if present.
# 3. **Feature engineering (customer-level)**
#    - RFM-like signals: `txn_count`, `days_active`, `total_sales_sum/mean/median`, `avg_basket_size`, `avg_unit_price`, `recency_days`.
#    - Compact preference profile: **Top-K (K=10) `product_subclass` proportions** per customer.
# 4. **Scaling & encoding (modeling matrices)**
#    - **Winsorize** numeric features at **99.5th percentile** (modeling copies only) to tame long tails.
#    - **Numerics:** median **imputation** → **StandardScaler** (zero-mean/unit-variance).
#    - **Categoricals:** **One-Hot Encoding** with `handle_unknown='ignore'` (e.g., `age_group`, `pin_code`, `day_of_week`, `year_month` for transactions).
#    - Outputs:
#      - `X_txn.npy` + `X_txn_feature_names.json` — transaction-level matrix & names
#      - `X_cust.npy` + `X_cust_feature_names.json` — customer-level matrix & names
# 5. **Save tidy references**
#    - `artifacts/preprocessed/transactions_clean.parquet` — minimally imputed facts  
#    - `artifacts/preprocessed/transactions_features.parquet` — transaction features  
#    - `artifacts/preprocessed/customers_features.parquet` — customer features (if applicable)
# 
# **Why this matters.**
# - EDA showed heavy skew & high cardinality; this pipeline stabilizes features for distance-based methods (k-means, PCA/UMAP) and preserves interpretable copies.
# - **Top-K subclass proportions** capture product-mix signals without exploding dimensionality.
# 
# **Knobs you can tweak.**
# - Winsorization level (`upper_q=0.995`) for tail strength.
# - Item granularity in EDA (`PRODUCT_ID` vs `PRODUCT_SUBCLASS`) for basket construction.
# - K for subclass proportions (default 10) based on variance explained or downstream performance.
# 

# In[ ]:


# === Data Preprocessing: missing handling, feature engineering, scaling (FIXED) ===
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ART_DIR = Path("artifacts")
EDA_DIR = ART_DIR / "eda"
OUT_DIR = ART_DIR / "preprocessed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load normalized transactions + baskets from EDA
df = pd.read_parquet(EDA_DIR / "transactions_normalized.parquet")
baskets = pd.read_parquet(EDA_DIR / "baskets.parquet")

# --- KEY FIX: normalize baskets column names & ensure basket_size exists ---
baskets = baskets.copy()
baskets.columns = [c.strip().lower() for c in baskets.columns]

# Ensure 'items' exists
assert "items" in baskets.columns, "baskets.parquet must contain an 'items' column."

# Add basket_size if absent
if "basket_size" not in baskets.columns:
    baskets["basket_size"] = baskets["items"].apply(lambda xs: len(set(xs)) if isinstance(xs, (list, tuple)) else 0)

# -----------------------------
# 1) Handling Missing Data
# -----------------------------
# Critical keys should not be missing
key_cols = ["transaction_dt"]
if "customer_id" in df.columns:
    key_cols.append("customer_id")

df = df.dropna(subset=key_cols).copy()

# AGE_GROUP: impute to 'Unknown' (~3% missing)
if "age_group" in df.columns:
    df["age_group"] = df["age_group"].fillna("Unknown").astype(str)

# PIN_CODE: fill blanks with 'Unknown'
if "pin_code" in df.columns:
    df["pin_code"] = df["pin_code"].fillna("Unknown").astype(str)

# Numeric coercion (re-assert)
for c in ["amount", "asset", "sales_price"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# 2) Feature Engineering
# -----------------------------
# Temporal features
df["year_month"] = df["transaction_dt"].dt.to_period("M").astype(str)
df["day_of_week"] = df["transaction_dt"].dt.day_name()
df["week"] = df["transaction_dt"].dt.to_period("W").astype(str)

# Transaction-level totals (sum over lines)
txn_key = ["customer_id", "transaction_dt"] if "customer_id" in df.columns else ["transaction_dt"]

agg_txn = df.groupby(txn_key).agg(
    total_amount=("amount", "sum"),
    total_asset=("asset", "sum"),
    total_sales=("sales_price", "sum"),
    n_lines=("product_id", "count") if "product_id" in df.columns else ("product_subclass", "count"),
).reset_index()

# Merge basket_size from baskets (distinct items)
# Align keys in baskets to match df (already lowercase)
needed_keys = [k for k in txn_key if k in baskets.columns]
assert set(needed_keys) == set(txn_key), (
    f"Key mismatch between df {txn_key} and baskets {list(baskets.columns)}. "
    "Ensure baskets has the same key columns."
)

tmp = baskets[txn_key + ["basket_size"]].copy()
agg_txn = agg_txn.merge(tmp, on=txn_key, how="left")

# Unit price proxy
agg_txn["avg_unit_price"] = np.where(agg_txn["total_amount"] > 0,
                                     agg_txn["total_sales"] / agg_txn["total_amount"],
                                     np.nan)

# Attach simple categorical/time context to transactions
ctx_cols = ["age_group","pin_code","year_month","day_of_week","week"]
ctx_merge = (df[txn_key + [c for c in ctx_cols if c in df.columns]]
               .drop_duplicates(subset=txn_key))
agg_txn = agg_txn.merge(ctx_merge, on=txn_key, how="left")

# -----------------------------
# 2b) Customer-level aggregates
# -----------------------------
cust_key = ["customer_id"] if "customer_id" in df.columns else None
if cust_key:
    cust = agg_txn.groupby("customer_id").agg(
        txn_count=("total_sales", "size"),
        days_active=("transaction_dt", lambda s: (s.max() - s.min()).days if s.notna().any() else 0),
        total_sales_sum=("total_sales", "sum"),
        total_sales_mean=("total_sales", "mean"),
        total_sales_median=("total_sales", "median"),
        avg_basket_size=("basket_size", "mean"),
        avg_unit_price=("avg_unit_price", "mean")
    ).reset_index()

    # Recency proxy
    end_date = df["transaction_dt"].max()
    last_txn = agg_txn.groupby("customer_id")["transaction_dt"].max().reset_index(name="last_txn_dt")
    last_txn["recency_days"] = (end_date - last_txn["last_txn_dt"]).dt.days
    cust = cust.merge(last_txn[["customer_id","recency_days"]], on="customer_id", how="left")

    # Subclass composition (top-K proportions)  ✅ FIXED (use transform, not apply)
    K = 10
    if "product_subclass" in df.columns:
        sub_pivot = (
            df.groupby(["customer_id", "product_subclass"])
              .size()
              .reset_index(name="cnt")
        )
        # per-customer totals with transform keeps the original index → safe to assign
        totals = sub_pivot.groupby("customer_id")["cnt"].transform("sum").replace(0, np.nan)
        sub_pivot["prop"] = sub_pivot["cnt"] / totals

        # pick global top-K subclasses to limit width
        global_topK = (
            sub_pivot.groupby("product_subclass")["cnt"].sum()
            .sort_values(ascending=False)
            .head(K)
            .index.tolist()
        )

        sub_top = (
            sub_pivot[sub_pivot["product_subclass"].isin(global_topK)]
            .pivot(index="customer_id", columns="product_subclass", values="prop")
            .fillna(0.0)
        )
        sub_top.columns = [f"subcls_prop_{c}" for c in sub_top.columns]
        cust = (
            cust.merge(sub_top, left_on="customer_id", right_index=True, how="left")
                .fillna(0.0)
        )

# -----------------------------
# 3) Standardisation / Normalisation
# -----------------------------
def winsorize_series(s, upper_q=0.995):
    q = s.quantile(upper_q)
    return s.clip(upper=q)

WINSORIZE_FOR_MODELING = True
NUM_TXN = ["total_amount","total_asset","total_sales","basket_size","n_lines","avg_unit_price"]
NUM_TXN = [c for c in NUM_TXN if c in agg_txn.columns]
CAT_TXN = [c for c in ["age_group","pin_code","day_of_week","year_month"] if c in agg_txn.columns]

if WINSORIZE_FOR_MODELING:
    for c in NUM_TXN:
        agg_txn[c] = winsorize_series(agg_txn[c])

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

txn_ct = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("impute", num_imputer), ("scale", StandardScaler())]),
         NUM_TXN),
        ("cat", Pipeline(steps=[("impute", cat_imputer), ("ohe", ohe)]),
         CAT_TXN),
    ],
    remainder="drop"
)

X_txn = txn_ct.fit_transform(agg_txn)
num_names = NUM_TXN
cat_names = list(txn_ct.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(CAT_TXN)) if CAT_TXN else []
txn_feature_names = num_names + cat_names

np.save(OUT_DIR / "X_txn.npy", X_txn)
pd.DataFrame(agg_txn[txn_key + NUM_TXN + CAT_TXN]).to_parquet(OUT_DIR / "transactions_features.parquet", index=False)
with open(OUT_DIR / "X_txn_feature_names.json", "w") as f:
    import json; json.dump(txn_feature_names, f, indent=2)

print(f"Transaction-level matrix: X_txn shape = {X_txn.shape}")

# Customer-level matrix
if cust_key:
    NUM_CUST = ["txn_count","days_active","total_sales_sum","total_sales_mean","total_sales_median",
                "avg_basket_size","avg_unit_price","recency_days"]
    NUM_CUST += [c for c in cust.columns if c.startswith("subcls_prop_")]
    if WINSORIZE_FOR_MODELING:
        for c in NUM_CUST:
            cust[c] = winsorize_series(cust[c])

    cust_ct = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]),
             NUM_CUST)
        ],
        remainder="drop"
    )
    X_cust = cust_ct.fit_transform(cust)
    np.save(OUT_DIR / "X_cust.npy", X_cust)
    cust.to_parquet(OUT_DIR / "customers_features.parquet", index=False)
    with open(OUT_DIR / "X_cust_feature_names.json", "w") as f:
        import json; json.dump(NUM_CUST, f, indent=2)
    print(f"Customer-level matrix: X_cust shape = {X_cust.shape}")

# -----------------------------
# 4) Save cleaned transactions (minimal imputation)
# -----------------------------
df.to_parquet(OUT_DIR / "transactions_clean.parquet", index=False)

print("\nSaved artifacts:")
print(f"- {OUT_DIR/'transactions_clean.parquet'}")
print(f"- {OUT_DIR/'transactions_features.parquet'}")
if cust_key:
    print(f"- {OUT_DIR/'customers_features.parquet'}")
print(f"- {OUT_DIR/'X_txn.npy'} (+ X_txn_feature_names.json)")
if cust_key:
    print(f"- {OUT_DIR/'X_cust.npy'} (+ X_cust_feature_names.json)")


# ## 3. Data Mining Methods and Analysis
# 
# **Setting.** Each transaction is a set of items with **existential probabilities** \(P_t(x)\in(0,1]\). We mine frequent itemsets under **expected support**:
# \[
# S_e(X)=\sum_{t\in D}\ \prod_{x\in X} P_t(x),
# \]
# and declare \(X\) frequent if \(S_e(X)\ge \theta\) (absolute minsup).
# 
# **Algorithms compared.**
# 1. **U-Apriori (Expected Support).** Level-wise candidate generation (Apriori-Gen) with expected-support counting (product of item probabilities within transactions).
# 2. **LGS-Trimming** (paper’s speedup for uncertain data):
#    - **Local trimming:** per-item threshold \(\rho_t(x)\) removes very small \(P_t(x)\) to form a trimmed dataset \(D_T\).
#    - **Global pruning:** compute an **upper bound** for the lost (trimmed-away) contribution; if \(S_e^T(X)+\text{UB}(X)<\theta\), prune.
#    - **Single-pass patch-up:** recompute exact expected support on the **original** (untrimmed) data for survivors and finalize frequency.
# 
# **Protocol.**
# - Build a **single probabilistic dataset** from the baskets (jitter probabilities for observed items and add small-probability noise items) and keep it fixed.
# - Run **U-Apriori** and **LGS-Trimming** with the **same minsup**.
# - Report runtime, number of frequent itemsets, and top-k itemsets by expected support.
# 

# ### 3.1 Config + probabilistic data (single source of truth)

# In[ ]:


# --- Probabilistic data per paper: high/low Gaussians + R control ---
from pathlib import Path
import pandas as pd
import numpy as np
import random

EDA_DIR = Path("artifacts/eda")
baskets = pd.read_parquet(EDA_DIR / "baskets.parquet").copy()
assert "items" in baskets, "baskets.parquet must contain an 'items' column."
N_TX = len(baskets)

# === Experiment knobs (tune to mirror the paper's sweeps) ===
REL_MINSUP = 0.005              # e.g., 0.5%  (you can sweep later: 1%, 0.5%, 0.2%, 0.1%)
ABS_MINSUP = REL_MINSUP * N_TX
TOPK_SHOW  = 20

# R = T_low / (T_high + T_low) → share of low-probability items in the dataset
R = 0.333                        # try {0.0, 0.333, 0.5, 0.667, 0.75}

# High/Low Gaussian params (means, stddevs); clip to [0,1]
HB, HD = 0.90, 0.05             # "high-prob" ~ N(0.90, 0.05^2)
LB, LD = 0.10, 0.05             # "low-prob"  ~ N(0.10, 0.05^2)

# Optional: add a small number of *extra* low-prob items per transaction
ADD_NOISE_LOW = False
NOISE_K = 3                      # # of extra items to inject per transaction
NOISE_P_IS_GAUSSIAN = True       # if True draw from N(LB, LD), else fixed LB

SEED = 7
rng_py = random.Random(SEED)
rng_np = np.random.default_rng(SEED)

ALL_ITEMS = sorted({it for xs in baskets["items"] for it in set(xs)})

def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def _draw_high(n):
    return np.clip(rng_np.normal(HB, HD, size=n), 0.0, 1.0).astype(float)

def _draw_low(n):
    return np.clip(rng_np.normal(LB, LD, size=n), 0.0, 1.0).astype(float)

def make_probabilistic_R_model(baskets, R, add_noise_low=False, noise_k=5, noise_gaussian=True):
    """
    Paper-style uncertainty:
      - Start from deterministic baskets (observed items).
      - Assign each observed item to 'low' with probability R and 'high' otherwise.
        Draw p from N(HB,HD) for high, N(LB,LD) for low; clip to [0,1].
      - Optionally add 'noise_k' extra low-probability items per transaction.

    Notes:
      - This achieves the target R approximately over the *observed* item occurrences.
      - Adding extra low-prob noise per txn increases the global low share slightly; keep noise_k small.
      - For exact global R matching, we could post-correct, but the paper's trends are robust to small deltas.
    """
    prob_txns = []
    low_count, total_count = 0, 0

    for items in baskets["items"]:
        items = list(set(map(str, items)))
        m = len(items)
        # Bernoulli assignment of low/high for observed items
        is_low = rng_np.random(m) < R
        ps = np.empty(m, dtype=float)
        # draw probs
        n_low = int(is_low.sum())
        n_high = m - n_low
        if n_high > 0:
            ps[~is_low] = _draw_high(n_high)
        if n_low > 0:
            ps[is_low] = _draw_low(n_low)

        tx = {it: _clip01(p) for it, p in zip(items, ps)}
        low_count += int(n_low); total_count += m

        # Optional: add extra low-prob items (noise)
        if add_noise_low and noise_k > 0:
            # sample without replacing existing items
            add_pool = [it for it in ALL_ITEMS if it not in tx]
            for _ in range(min(noise_k, len(add_pool))):
                it = rng_py.choice(add_pool)
                add_pool.remove(it)
                if noise_gaussian:
                    p = float(_draw_low(1)[0])
                else:
                    p = float(LB)
                tx[it] = _clip01(p)
                low_count += 1; total_count += 1

        prob_txns.append(tx)

    achieved_R = low_count / max(1, total_count)
    print(f"Target R={R:.3f} | Achieved R≈{achieved_R:.3f} "
          f"(low={low_count}, total={total_count})")
    return prob_txns

# Build the canonical uncertain dataset for ALL methods (U-Apriori & LGS)
prob_txns = make_probabilistic_R_model(
    baskets,
    R=R,
    add_noise_low=ADD_NOISE_LOW,
    noise_k=NOISE_K,
    noise_gaussian=NOISE_P_IS_GAUSSIAN
)

print(f"Transactions: {N_TX} | rel minsup={REL_MINSUP:.3%} | abs minsup={ABS_MINSUP:.2f}")
print("Example txn (first 5 items):", list(prob_txns[0].items())[:5])


# ### 3.2 U-Apriori (expected support)

# In[ ]:


# --- U-Apriori (Expected Support) ---
from itertools import combinations
from collections import Counter, defaultdict
import time

def apriori_gen(Lk_1):
    """
    Join frequent (k-1)-itemsets (as frozensets) to form size-k candidates (as frozensets).
    Correctly checks that all (k-1)-subsets of a candidate are in Lk_1.
    """
    # normalize: list of sorted tuples just for deterministic joining
    L = sorted([tuple(sorted(x)) for x in Lk_1])
    if not L:
        return set()
    k = len(L[0]) + 1
    Ck = set()
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            a, b = L[i], L[j]
            # join on common prefix (length k-2)
            if a[:-1] == b[:-1]:
                cand = tuple(sorted(a + (b[-1],)))          # tuple for deterministic order
                # ✅ fix: check presence of all (k-1)-subsets in Lk_1 using frozenset
                if all(frozenset(s) in Lk_1 for s in combinations(cand, k - 1)):
                    Ck.add(frozenset(cand))                  # store as frozenset
    return Ck


def expected_support_k(prob_txns, Ck):
    Se = {c: 0.0 for c in Ck}
    for tx in prob_txns:
        txset = set(tx.keys())
        cand_here = [c for c in Ck if c.issubset(txset)]
        for c in cand_here:
            prod = 1.0
            for it in c:
                prod *= tx[it]
            Se[c] += prod
    return Se

def u_apriori(prob_txns, abs_minsup):
    t0 = time.time()
    # 1-itemsets
    Se1 = Counter()
    for tx in prob_txns:
        for it, p in tx.items():
            Se1[frozenset([it])] += p
    L1 = {c for c,se in Se1.items() if se >= abs_minsup}
    Se_map = dict(Se1)
    levels = [L1]

    while levels[-1]:
        Ck = apriori_gen(levels[-1])
        if not Ck: break
        Sek = expected_support_k(prob_txns, Ck)
        Lk  = {c for c,se in Sek.items() if se >= abs_minsup}
        Se_map.update(Sek)
        levels.append(Lk)

    F = set().union(*levels) if levels else set()
    secs = time.time() - t0
    return F, Se_map, secs

F_u, Se_u, secs_u = u_apriori(prob_txns, ABS_MINSUP)
print(f"U-Apriori: |F|={len(F_u)} | time={secs_u:.2f}s")


# ### 3.3 LGS-Trimming (local trim → global prune → patch-up)

# In[ ]:


# --- LGS-Trimming pipeline ---
import numpy as np
from collections import defaultdict
import time

def collect_item_probs(prob_txns):
    obs = defaultdict(list)
    for tx in prob_txns:
        for it, p in tx.items():
            obs[it].append(p)
    return obs

def choose_local_thresholds(item_obs, abs_minsup, early_frac=0.20):
    """
    Local trimming thresholds ρ_t(x) per item x.

    For each item x:
      1) Sort observed existential probabilities p_i in descending order.
      2) Compute cumulative mass c_k = sum_{i<=k} p_i.
      3) p1 (crossing): first index k where c_k >= abs_minsup; threshold = p_k.
      4) p2 (knee): normalize x = k/n, y = c_k / c_n and choose k maximizing (y - x).
         (This is a robust elbow/knee heuristic for concave cumulative curves.)
      5) Decision:
           - If crossing exists and happens "early" (k/n <= early_frac), use p1.
           - Else use p2.
    Edge cases:
      - No observations: ρ = 1.0 (effectively keep nothing).
      - Flat/monotone mass (no knee): fallback to mid index or last prob > 0.
    """
    rho_t = {}
    for it, plist in item_obs.items():
        if not plist:
            rho_t[it] = 1.0
            continue

        # 1) sort probs descending
        l = np.array(sorted(plist, reverse=True), dtype=float)
        n = len(l)

        # Guard: if all zeros, set ρ high
        if np.all(l <= 0):
            rho_t[it] = 1.0
            continue

        # 2) cumulative mass
        c = np.cumsum(l)
        total = c[-1]

        # 3) p1 = first crossing of abs_minsup, if any
        p1_idx = None
        if abs_minsup > 0:
            hit = np.where(c >= abs_minsup)[0]
            if len(hit) > 0:
                p1_idx = int(hit[0])
        p1_thr = l[p1_idx] if p1_idx is not None else None

        # 4) p2 = knee via max(y - x) on normalized cumulative curve
        #    x in [1/n, ..., 1], y in [c1/total, ..., 1]
        #    (y - x) peaks at the elbow for concave curves.
        x = (np.arange(1, n + 1) / n)
        y = c / max(total, 1e-12)
        diff = y - x
        knee_idx = int(np.argmax(diff))  # robust elbow index
        p2_thr = float(l[knee_idx])

        # 5) decide regime
        use_p1 = False
        if p1_idx is not None:
            frac = (p1_idx + 1) / n  # crossing position as a fraction
            if frac <= early_frac:
                use_p1 = True

        rho_t[it] = float(p1_thr) if use_p1 and (p1_thr is not None) else float(p2_thr)

        # Safety: ensure threshold ∈ (0,1] and not greater than max observed
        rho_t[it] = max(0.0, min(rho_t[it], float(l[0])))

    return rho_t


def build_trimmed_dataset(prob_txns, rho_t):
    DT = []
    ST_e, SnotT_e = defaultdict(float), defaultdict(float)
    MT, MnotT     = defaultdict(float), defaultdict(float)
    for tx in prob_txns:
        kept = {}
        for it, p in tx.items():
            if p >= rho_t.get(it, 1.0):
                kept[it] = p
                ST_e[it] += p
                MT[it] = max(MT[it], p)
            else:
                SnotT_e[it] += p
                MnotT[it] = max(MnotT[it], p)
        DT.append(kept)
    return DT, dict(ST_e), dict(SnotT_e), dict(MT), dict(MnotT)

def expected_support_k_on_DT(DT, Ck):
    Se = {c: 0.0 for c in Ck}
    for tx in DT:
        txset = set(tx.keys())
        cand_here = [c for c in Ck if c.issubset(txset)]
        for c in cand_here:
            prod = 1.0
            for it in c:
                prod *= tx[it]
            Se[c] += prod
    return Se

def _safe_div(num, den):
    return float(num) / float(den) if den and den > 0 else 0.0

def _term_ST_notT_AB(A, B, ST_e_item, SnotT_e_item, MT_item, MnotT_item, ST_e_AB):
    # Equation (5): Ŝ^{T,~T}_e(AB)
    MT_A   = MT_item.get(A, 0.0)
    MnotT_B= MnotT_item.get(B, 0.0)
    if MT_A == 0 or MnotT_B == 0:
        return 0.0
    ST_e_A = ST_e_item.get(A, 0.0)
    SnotT_e_B = SnotT_e_item.get(B, 0.0)
    cap_left  = _safe_div( max(0.0, ST_e_A - ST_e_AB), MT_A )
    cap_right = _safe_div( SnotT_e_B, MnotT_B )
    return min(cap_left, cap_right) * MT_A * MnotT_B

def _term_notT_ST_AB(A, B, ST_e_item, SnotT_e_item, MT_item, MnotT_item, ST_e_AB):
    # Equation (6): Ŝ^{~T,T}_e(AB)
    MT_B   = MT_item.get(B, 0.0)
    MnotT_A= MnotT_item.get(A, 0.0)
    if MT_B == 0 or MnotT_A == 0:
        return 0.0
    SnotT_e_A = SnotT_e_item.get(A, 0.0)
    ST_e_B    = ST_e_item.get(B, 0.0)
    cap_left  = _safe_div( SnotT_e_A, MnotT_A )
    cap_right = _safe_div( max(0.0, ST_e_B - ST_e_AB), MT_B )
    return min(cap_left, cap_right) * MnotT_A * MT_B

def _term_notT_notT_AB(A, B, SnotT_e_item, MnotT_item, term_ST_notT, term_notT_ST):
    # Equation (7): Ŝ^{~T,~T}_e(AB)
    MnotT_A = MnotT_item.get(A, 0.0)
    MnotT_B = MnotT_item.get(B, 0.0)
    if MnotT_A == 0 or MnotT_B == 0:
        return 0.0
    SnotT_e_A = SnotT_e_item.get(A, 0.0)
    SnotT_e_B = SnotT_e_item.get(B, 0.0)
    cap_left  = _safe_div( max(0.0, SnotT_e_A - term_notT_ST), MnotT_A )
    cap_right = _safe_div( max(0.0, SnotT_e_B - term_ST_notT), MnotT_B )
    return min(cap_left, cap_right) * MnotT_A * MnotT_B

def global_prune_upper_bound_pair(X, ST_e_item, SnotT_e_item, MT_item, MnotT_item, ST_e_AB):
    """
    Exact global error bound for pairs {A,B} per the paper:
      e(AB) = Ŝ^{T,~T}_e + Ŝ^{~T,T}_e + Ŝ^{~T,~T}_e
    Uses Equations (5), (6), (7). If |X| != 2, returns None (caller can fallback).
    """
    if len(X) != 2:
        return None
    A, B = tuple(sorted(X))
    term1 = _term_ST_notT_AB(A, B, ST_e_item, SnotT_e_item, MT_item, MnotT_item, ST_e_AB)
    term2 = _term_notT_ST_AB(A, B, ST_e_item, SnotT_e_item, MT_item, MnotT_item, ST_e_AB)
    term3 = _term_notT_notT_AB(A, B, SnotT_e_item, MnotT_item, term1, term2)
    return term1 + term2 + term3


def lgs_trimming(prob_txns, abs_minsup):
    import time
    from collections import defaultdict, Counter

    t0 = time.time()

    # --- Local thresholds & trimmed dataset ---
    item_obs = collect_item_probs(prob_txns)
    rho_t    = choose_local_thresholds(item_obs, abs_minsup)
    DT, ST_e, SnotT_e, MT, MnotT = build_trimmed_dataset(prob_txns, rho_t)

    # --- k=1 on DT ---
    Se1 = Counter()
    for tx in DT:
        for it, p in tx.items():
            Se1[frozenset([it])] += p

    # Items frequent on DT (used for joining)
    L1_DT = {c for c, se in Se1.items() if se >= abs_minsup}

    # ✅ Singletons that are infrequent on DT but frequent overall.
    # These are for FINAL OUTPUT ONLY, not for joining.
    L1_extra_output_only = set()
    all_items = set(ST_e.keys()) | set(SnotT_e.keys())
    for it in all_items:
        se_orig = ST_e.get(it, 0.0) + SnotT_e.get(it, 0.0)
        if se_orig >= abs_minsup:
            L1_extra_output_only.add(frozenset([it]))

    # Use only DT-frequent items to seed joining (keeps C2 manageable)
    levels = [L1_DT]
    Se_DT = dict(Se1)

    # --- k >= 2 on DT with exact pair upper bound; conservative for k>2 ---
    while levels[-1]:
        Ck = apriori_gen(levels[-1])
        if not Ck:
            break

        Sek_DT = expected_support_k_on_DT(DT, Ck)

        # Frequent on DT outright
        survivors = {c for c, se in Sek_DT.items() if se >= abs_minsup}

        # Infrequent on DT: check global UB (exact for pairs)
        for X, se_dt in Sek_DT.items():
            if se_dt >= abs_minsup:
                continue
            UB_pair = global_prune_upper_bound_pair(X, ST_e, SnotT_e, MT, MnotT, ST_e_AB=se_dt)
            if UB_pair is None:
                # For k>2 we keep conservative behavior: no global pruning here.
                continue
            if se_dt + UB_pair >= abs_minsup:
                survivors.add(X)

        Se_DT.update(Sek_DT)
        levels.append(survivors)

    # --- Patch-up on original probabilistic data for final exact Se ---
    candidates = set().union(*levels) if levels else set()

    # Ensure singletons frequent overall are included in FINAL OUTPUT
    candidates |= L1_extra_output_only

    Se_true = {c: 0.0 for c in candidates}
    for tx in prob_txns:
        txset = set(tx.keys())
        for c in candidates:
            if c.issubset(txset):
                prod = 1.0
                for it in c:
                    prod *= tx[it]
                Se_true[c] += prod

    F_final = {c for c, se in Se_true.items() if se >= abs_minsup}
    secs = time.time() - t0
    return F_final, Se_true, secs, rho_t, DT



F_lgs, Se_lgs, secs_lgs, rho_t, DT = lgs_trimming(prob_txns, ABS_MINSUP)
print(f"LGS-Trimming: |F|={len(F_lgs)} | time={secs_lgs:.2f}s")


# ### 3.4 Head-to-head summary + top-k itemsets

# In[ ]:


import pandas as pd

print("\n=== Runtime (seconds) ===")
print(f"U-Apriori     : {secs_u:.2f}")
print(f"LGS-Trimming  : {secs_lgs:.2f}")

print("\n=== # Frequent Itemsets ===")
print(f"U-Apriori |F| : {len(F_u)}")
print(f"LGS      |F|  : {len(F_lgs)}")

def top_itemsets(Se_map, F, k=TOPK_SHOW):
    rows = sorted([(tuple(sorted(fs)), Se_map.get(fs, 0.0)) for fs in F], key=lambda x: -x[1])[:k]
    return pd.DataFrame(rows, columns=["itemset","expected_support"])

print("\nTop-k itemsets by expected support (U-Apriori):")
display(top_itemsets(Se_u, F_u, k=15))

print("\nTop-k itemsets by expected support (LGS-Trimming):")
display(top_itemsets(Se_lgs, F_lgs, k=15))

