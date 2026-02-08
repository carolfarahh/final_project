import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import statsmodels.api as sm
from src.app_logger import logger



#####EDA###
#Boxplot of Brain Volume Loss across Disease Stages This plot helps visualize differences in 
# the distribution of Brain_Volume_Loss between disease stages (median, spread, and potential outliers).
def eda_plots(df):
    plt.figure()
    df.boxplot(column="Brain_Volume_Loss", by="Disease_Stage")
    plt.title("Brain_Volume_Loss by Disease_Stage")
    plt.suptitle("")  # removes the automatic pandas subtitle
    plt.xlabel("Disease_Stage")
    plt.ylabel("Brain_Volume_Loss")
    plt.xticks(rotation=20)
    plt.show()
    #Histogram of Brain Volume Loss This plot shows 
    #the overall distribution of Brain_Volume_Loss to check skewness and potential outliers.
    plt.figure()
    plt.hist(df["Brain_Volume_Loss"], bins=30)
    plt.title("Distribution of Brain_Volume_Loss")
    plt.xlabel("Brain_Volume_Loss")
    plt.ylabel("Count")
    plt.show()
    #Scatter plot (Age vs Brain Volume Loss) with transparency This scatter plot visualizes 
    # the relationship between Age and Brain_Volume_Loss. Because the dataset is large, many 
    # points overlap; therefore, we use transparency (alpha) to reduce overplotting and make the density of points easier to interpret.
    plt.figure()
    plt.scatter(df["Age"], df["Brain_Volume_Loss"], s=8, alpha=0.2)
    plt.title("Age vs Brain_Volume_Loss (alpha=0.2)")
    plt.xlabel("Age")
    plt.ylabel("Brain_Volume_Loss")
    plt.show()
    #Boxplot of Brain Volume Loss by Sex This plot compares the distribution of Brain_Volume_Loss between sexes 
    #(median and spread), which supports the “adjust for Sex” part of the research question.
    plt.figure()
    df.boxplot(column="Brain_Volume_Loss", by="Sex")
    plt.title("Brain_Volume_Loss by Sex")
    plt.suptitle("")
    plt.xlabel("Sex")
    plt.ylabel("Brain_Volume_Loss")
    plt.show()

###Statistical Assumptions####
def linearity_graph_cov_dv(x,y,linearity_r, linearity_p, dv, cov, show_plot=None, kind="hexbin"):

        if show_plot == True:
            plt.figure(figsize=(8, 5))
        
            if kind == "scatter":
                # s=1 makes dots tiny; alpha adds transparency to show density
                plt.scatter(x, y, s=1, alpha=0.3, edgecolors='none', color='steelblue')
            elif kind == "hexbin":
                # Instead of drawing 10,000 dots, it divides the plot into a honeycomb of hexagons
                # Bu-Pu a color gradient where light colors represent few points 
                # and dark purple represents high density
                hb = plt.hexbin(x, y, gridsize=50, cmap='BuPu', mincnt=1)

                #Labels the column of the heat map on the side with "Count"
                plt.colorbar(hb, label='Count')

            # best-fit line (linear line)
            m, b = np.polyfit(x, y, 1) # Conducts least squares
            x_line = np.linspace(x.min(), x.max(), 200) #Defines boundaries of linear line
            plt.plot(x_line, m * x_line + b, color='darkorange', linewidth=2, label='Best Fit') #Creates formula and plots the linear line

            plt.xlabel(cov) #X-axis label
            plt.ylabel(dv) #Y-axis label
            plt.title(f"Linearity: {cov} vs {dv}\n(r={linearity_r:.2f}, p={linearity_p:.3g})") #graph title
            plt.legend()  #helps  identify and distinguish between different data series and lines
            plt.show() #Shows the plot


def check_normality_of_residuals_visual(df,dv,iv,covariate):
    """
    To check for normality of residuals, we have to create a graph for all of the residuals
    and physically check if the residuals are distributed linearly.
    In this function, we create two graphs; Q-Q plot and Histogram, using matplotlib.pyplot library.

    """

    model = ols(f"{dv} ~ C({iv}) * {covariate}", data=df).fit() #We fit the model of ANCOVA

    # We create an array of the residuals using the .resid function
    # Just in case some of the residuals had NaN values we drop them to ensure smoother output of graphs
    resid = model.resid.dropna() 

    # Histogram
    plt.figure()
    plt.hist(resid, bins=30) #It controls the granularity of the plot
    plt.title("Residuals Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

    # Q-Q Plot
    plt.figure()
    sm.qqplot(resid, line="45") #Adds a 45-degree reference line for comparrison 
    plt.title("Q-Q Plot of Residuals")
    plt.show()

###########

#ANOVA graph

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



#ANCOVA 
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

#Moderated regression
def adjusted_means_plot_moderated(df, dv, iv, moderator):

    # Fit moderated regression model
    model = smf.ols(f"{dv} ~ C({iv}) * {moderator}", data=df).fit()

    # Define moderator levels: mean, +/- 1 SD
    mod_mean = df[moderator].mean()
    mod_sd = df[moderator].std()
    mod_levels = [mod_mean - mod_sd, mod_mean, mod_mean + mod_sd]
    mod_labels = ["-1 SD", "Mean", "+1 SD"]

    # Get unique IV levels
    iv_levels = df[iv].unique()

    # Build prediction table
    pred_list = []
    for m in mod_levels:
        for iv_level in iv_levels:
            pred_list.append({iv: iv_level, moderator: m})
    pred_df = pd.DataFrame(pred_list)

    # Get predicted means + CI
    pred = model.get_prediction(pred_df).summary_frame()
    pred_df["mean"] = pred["mean"]
    pred_df["low"] = pred["mean_ci_lower"]
    pred_df["high"] = pred["mean_ci_upper"]
    pred_df["mod_label"] = np.repeat(mod_labels, len(iv_levels))

    # Plot
    plt.figure(figsize=(8, 5))
    for m_label in mod_labels:
        df_plot = pred_df[pred_df["mod_label"] == m_label]
        x = np.arange(len(iv_levels))
        plt.errorbar(
            x, df_plot["mean"], 
            yerr=[df_plot["mean"] - df_plot["low"], df_plot["high"] - df_plot["mean"]],
            fmt="o-", capsize=5, label=m_label
        )
    plt.xticks(np.arange(len(iv_levels)), iv_levels)
    plt.xlabel(iv)
    plt.ylabel(f"Predicted {dv}")
    plt.title(f"Adjusted Means by {iv} at Moderator Levels ({moderator})")
    plt.legend(title=moderator)
    plt.show()

    



    
