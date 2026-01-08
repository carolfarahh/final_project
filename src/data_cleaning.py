import pandas as pd

def filter_stage_gene(
    df: pd.DataFrame,
    stage="Pre-Symptomatic",
    genes=("MLH1", "MSH3", "HTT (Somatic Expansion)"),
    stage_col="Disease_Stage",
    gene_col="Gene/Factor",
):
    # validate columns
    missing = [c for c in (stage_col, gene_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}. Available: {list(df.columns)}")

    filtered_df = df.copy()

    # light standardization for safer matching
    filtered_df[stage_col] = filtered_df[stage_col].astype(str).str.strip()
    filtered_df[gene_col]  = filtered_df[gene_col].astype(str).str.strip()

    stage_value = str(stage).strip().lower()
    genes_values = [str(gene).strip().lower() for gene in genes]

    filtered_df = filtered_df[filtered_df[stage_col].str.lower() == stage_value]
    filtered_df = filtered_df[filtered_df[gene_col].str.lower().isin(genes_values)]

    return filtered_df.reset_index(drop=True)
