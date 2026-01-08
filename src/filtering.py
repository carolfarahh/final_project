def filter_pre_gene(df):
    wanted_stage = "Pre-Symptomatic"
    wanted_genes = ["MLH1", "MSH3", "HTT (Somatic Expansion)"]

    out = df.copy()

    out["Disease_Stage"] = out["Disease_Stage"].astype(str).str.strip()
    out["Gene/Factor"]   = out["Gene/Factor"].astype(str).str.strip()

    out = out[out["Disease_Stage"] == wanted_stage]
    out = out[out["Gene/Factor"].isin(wanted_genes)]

    return out.reset_index(drop=True)