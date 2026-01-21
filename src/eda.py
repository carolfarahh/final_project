import pandas as pd
import matplotlib.pyplot as plt


def _ensure_columns_exist(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def missing_summary(df, columns):
    _ensure_columns_exist(df, columns)
    n = len(df)
    rows = []
    for col in columns:
        miss = int(df[col].isna().sum())
        pct = (miss / n) * 100 if n > 0 else float("nan")
        rows.append({"column": col, "missing_count": miss, "missing_percent": pct})
    return pd.DataFrame(rows)


def group_counts(df, group_cols):
    _ensure_columns_exist(df, group_cols)
    out = (
        df.groupby(list(group_cols), dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return out


def numeric_summary(df, columns):
    _ensure_columns_exist(df, columns)
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "n": int(s.notna().sum()),
                "mean": float(s.mean()) if s.notna().any() else float("nan"),
                "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else float("nan"),
                "min": float(s.min()) if s.notna().any() else float("nan"),
                "q1": float(s.quantile(0.25)) if s.notna().any() else float("nan"),
                "median": float(s.median()) if s.notna().any() else float("nan"),
                "q3": float(s.quantile(0.75)) if s.notna().any() else float("nan"),
                "max": float(s.max()) if s.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def descriptives_by_groups(df, dv, group_cols, extra_means=None):
    cols_needed = [dv, *group_cols]
    if extra_means:
        cols_needed.extend(list(extra_means))
    _ensure_columns_exist(df, cols_needed)

    tmp = df.copy()
    tmp[dv] = pd.to_numeric(tmp[dv], errors="coerce")

    agg = {
        "n": (dv, lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        "mean": (dv, "mean"),
        "std": (dv, "std"),
        "median": (dv, "median"),
        "min": (dv, "min"),
        "max": (dv, "max"),
    }

    if extra_means:
        for col in extra_means:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            agg[f"mean_{col}"] = (col, "mean")

    out = (
        tmp.groupby(list(group_cols), dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(list(group_cols))
        .reset_index(drop=True)
    )
    return out


def plot_boxplot_stage_gene(df, dv, stage_col, gene_col):
    _ensure_columns_exist(df, [dv, stage_col, gene_col])
    tmp = df.copy()
    tmp[dv] = pd.to_numeric(tmp[dv], errors="coerce")

    stages = list(pd.Series(tmp[stage_col].dropna().unique()).astype(str))
    genes = list(pd.Series(tmp[gene_col].dropna().unique()).astype(str))

    data = []
    labels = []
    for st in stages:
        for g in genes:
            vals = tmp.loc[
                (tmp[stage_col].astype(str) == st) & (tmp[gene_col].astype(str) == g),
                dv,
            ].dropna()
            data.append(vals.values)
            labels.append(f"{st}\n{g}")

    fig, ax = plt.subplots()
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(dv)
    ax.set_title(f"{dv} by {stage_col} and {gene_col}")
    fig.tight_layout()
    return ax


def plot_means_lines_stage_gene(df, dv, stage_col, gene_col):
    _ensure_columns_exist(df, [dv, stage_col, gene_col])
    tmp = df.copy()
    tmp[dv] = pd.to_numeric(tmp[dv], errors="coerce")

    means = (
        tmp.groupby([stage_col, gene_col], dropna=False)[dv]
        .mean()
        .reset_index(name="mean_dv")
    )

    stages = list(pd.Series(means[stage_col].dropna().unique()).astype(str))
    genes = list(pd.Series(means[gene_col].dropna().unique()).astype(str))

    fig, ax = plt.subplots()
    x = list(range(len(stages)))

    for g in genes:
        y = []
        for st in stages:
            m = means.loc[
                (means[stage_col].astype(str) == st) & (means[gene_col].astype(str) == g),
                "mean_dv",
            ]
            y.append(float(m.iloc[0]) if len(m) else float("nan"))
        ax.plot(x, y, marker="o", label=str(g))

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_xlabel(stage_col)
    ax.set_ylabel(f"Mean {dv}")
    ax.set_title(f"Mean {dv} across {stage_col} (lines = {gene_col})")
    ax.legend()
    fig.tight_layout()
    return ax


def plot_scatter_age_vs_dv(df, age_col, dv, hue_col=None):
    cols = [age_col, dv] + ([hue_col] if hue_col else [])
    _ensure_columns_exist(df, cols)

    tmp = df.copy()
    tmp[age_col] = pd.to_numeric(tmp[age_col], errors="coerce")
    tmp[dv] = pd.to_numeric(tmp[dv], errors="coerce")
    tmp = tmp.dropna(subset=[age_col, dv])

    fig, ax = plt.subplots()

    if hue_col is None:
        ax.scatter(tmp[age_col], tmp[dv], s=10)
    else:
        cats = tmp[hue_col].astype(str)
        for cat in pd.unique(cats):
            sub = tmp.loc[cats == cat]
            ax.scatter(sub[age_col], sub[dv], s=10, label=cat)
        ax.legend()

    ax.set_xlabel(age_col)
    ax.set_ylabel(dv)
    ax.set_title(f"{dv} vs {age_col}" + (f" (by {hue_col})" if hue_col else ""))
    fig.tight_layout()
    return ax
