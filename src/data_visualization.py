

import matplotlib.pyplot as plt
import seaborn as sns

def boxplot_two_factor(df, dv, factor_a, factor_b,
                       title=None, ylabel=None, xlabel=None,
                       figsize=(8,6), palette="Set2"):
    
    # Check columns exist
    for col in [dv, factor_a, factor_b]:
        if col not in df.columns:
            raise KeyError(f"Column not found: {col}")

    # Plot
    plt.figure(figsize=figsize)
    sns.boxplot(x=factor_a, y=dv, hue=factor_b, data=df, palette=palette)
    
    plt.title(title or f"{dv} by {factor_a} and {factor_b}")
    plt.xlabel(xlabel or factor_a)
    plt.ylabel(ylabel or dv)
    plt.legend(title=factor_b)
    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ancova_linearity(df, dv="Brain_Volume_Loss", iv="Disease_Stage", cov="Age"):
    """
    Visualizes the relationship between a covariate (Age) and DV (Brain Volume)
    broken down by IV groups (Disease Stage).
    """
    plt.figure(figsize=(10, 6))
    
    # Use Seaborn's lmplot to handle the 10k points and multiple regression lines
    # scatter_kws={'s': 1} makes the 10k dots tiny
    # alpha=0.3 helps with overlap density
    sns.scatterplot(
        data=df, 
        x=cov, 
        y=dv, 
        hue=iv, 
        s=2, 
        alpha=0.3, 
        edgecolor=None,
        palette="viridis"
    )
    
    # Adding separate regression lines for each group to check for "Parallel Slopes"
    # This is the visual check for the 'homogeneity of regression slopes' assumption.
    unique_stages = df[iv].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_stages)))
    
    for stage, color in zip(unique_stages, colors):
        subset = df[df[iv] == stage]
        if len(subset) > 1:
            m, b = np.polyfit(subset[cov], subset[dv], 1)
            x_range = np.array([subset[cov].min(), subset[cov].max()])
            plt.plot(x_range, m * x_range + b, color=color, lw=2, label=f'Fit: {stage}')

    plt.title(f"ANCOVA Check: {dv} vs {cov} by {iv}")
    plt.xlabel(f"{cov} (Covariate)")
    plt.ylabel(f"{dv} (Dependent Variable)")
    plt.legend(title=iv, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# To use it:
# plot_ancova_linearity(your_dataframe)

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def adjusted_means_plot(df,dv, iv, cov):
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
