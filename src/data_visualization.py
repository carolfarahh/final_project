

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