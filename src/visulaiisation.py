#Adjusted means. (ANCOVA)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def adjusted_means_plot(df, IV, Covariate):
    # Fit ANCOVA model
    model = smf.ols(f"{dv} ~ C({iv}) + {cov}", data=df).fit()

    # Build prediction table (mean age for all stages)
    age_mean = df[cov].mean()
    stages = df[iv].unique()
    pred_df = pd.DataFrame({iv: stages, cov: age_mean})

    # Get adjusted means + CI
    pred = model.get_prediction(pred_df).summary_frame()
    means = pred["mean"].values
    low = pred["mean_ci_lower"].values
    high = pred["mean_ci_upper"].values

    # Plot
    x = np.arange(len(stages))
    plt.figure(figsize=(7, 4))
    plt.errorbar(x, means, yerr=[means-low, high-means], fmt="o", capsize=5)
    plt.xticks(x, stages)
    plt.xlabel(iv)
    plt.ylabel(f"Adjusted mean {dv}")
    plt.title(f"Adjusted Means by {iv} (Age fixed at mean={age_mean:.2f})")
    plt.show()

    return pred_df.assign(adjusted_mean=means, ci_low=low, ci_high=high)


#Box_Plot(two_way_ANOVA)
import matplotlib.pyplot as plt

def two_way_boxplot(df, dv="Brain_Volume_Loss", factor1="Disease_Stage", factor2="Sex"):
    groups = []
    labels = []

    for a in sorted(df[factor1].dropna().unique()):
        for b in sorted(df[factor2].dropna().unique()):
            vals = df[(df[factor1] == a) & (df[factor2] == b)][dv].dropna().astype(float).values
            if len(vals) > 0:
                groups.append(vals)
                labels.append(f"{a}-{b}")

    plt.figure(figsize=(10, 5))
    plt.boxplot(groups, labels=labels, showfliers=True)
    plt.xlabel(f"{factor1} × {factor2}")
    plt.ylabel(dv)
    plt.title(f"Box Plot of {dv} by {factor1} and {factor2}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
