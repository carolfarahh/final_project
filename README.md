# Demographic Moderation of Brain Volume Loss in Huntington’s Disease  
### A Statistical Analysis of Somatic Expansion Effects
## Authors
- Amane Egbaria
- Adi Drawshey
- Carol Farah

## Project Description
This project investigates how **demographic variables (age and sex)** moderate **brain volume loss** among patients with **Huntington’s disease (HD)** in the different stages of the disease. 

Huntington’s disease is a genetic neurodegenerative disorder caused by an expansion of CAG repeats in the *HTT* gene, leading to progressive motor, cognitive, and structural brain deterioration.
Meanwhile somatic expansion creates significantly larger, more toxic expansions (100–500+ CAGs) in brain cells, accelerating degeneration. Since our dataset had information about the gene factor, we though it would be best to explore somatic expansion in huntington on its own.

The analysis applies a rigorous statistical framework combining exploratory data analysis, regression-based modeling, and factorial designs to assess whether age or sex significantly influence brain volume loss beyond disease stage and genetic instability.
## Research Question
**How do demographics (age and sex) impact brain volume loss among Huntington’s disease patients with somatic expansion in different disease stages?**
## Hypotheses
1. **Age Effect:**  
   Age will have a significant effect on brain volume loss among Huntington’s disease patients with somatic expansion, such that older patients will exhibit greater brain volume loss.

2. **Sex Effect:**  
   Sex will not have a significant effect on brain volume loss among Huntington’s disease patients with somatic expansion.
## Assumptions
- Observations are independent (each patient appears once).
- The dataset is free of missing values in required analytical columns after cleaning.
- Outliers may bias inference and therefore must be detected and removed.
- Parametric test assumptions (linearity, homogeneity of variance, normality of residuals) are explicitly tested and addressed when violated.
## Data Description
- **Dataset Size (raw):** 48,536 rows × 21 columns  
- **Dataset Size (after cleaning):** 36,307 rows × 6 columns  
- **Missing Values:** 0% in required columns  
- **Duplicates:** 0  

### Selected Variables

**Demographics**
- `Age`
- `Sex`

**Clinical**
- `Brain_Volume_Loss`
- `Disease_Stage` (pre, early, middle, late)

**Genetic**
- `Gene/Factor`
  - MLH1
  - MSH3
  - HTT (Somatic Expansion)
## Data Cleaning & Validation
The data preprocessing pipeline was designed to ensure reliability, consistency, and statistical validity before analysis.
The following steps were applied:
- Selection of relevant columns and creation of a safe working copy.
- String normalization (case normalization and whitespace trimming).
- Filtering rows to include only relevant Gene/Factor values (MLH1, MSH3, HTT – Somatic Expansion).
- Conversion of numeric columns with invalid values coerced to NaN.
- Removal of rows with missing values in required fields.
- Removal of extreme values to reduce the influence of outliers.
- Sanity checks to ensure:
  - All required columns exist.
  - Categorical variables contain only allowed values.
  - Disease stage categories are valid.

If any validation step failed, the analysis was stopped immediately (fail-fast approach).
## Folder / Module Structure
The project is organized in a modular structure to ensure clarity, scalability, and reproducibility.

```text
final_project/
├── Huntington_Disease_Dataset.csv
├── main.py
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── data_import.py
│   ├── data_visualization.py
│   ├── data_cleaning.py
│   ├── data_import.py
│   ├── EDA.py
│   ├── statistical_analysis.py
│   ├── statistical_assumptions.py
│   └── validation.py
└── tests/
    ├── __init__.py
    ├── assumption_testing.py.py
    ├── EDA_testing.py
    ├── test_data_cleaning.py
    ├── test_data_import.py
    ├── test_statistical_analysis.py
    └── validation_tedting.py

```
## Validations pipelines:
- For each statistical tests, there were structural requirements that were needed to be met.
- A pipeline was constructed for each type of statistical test
- If the criterias weren't met, an error would be raised
- They criterias consisted of:
   -  Minimal levels in each category
   -  Minimal number of observations per group
   -  Type of Variables (Categorical or continuous)
   -  Dropping missing values for sanity checks etc
  
## Assumptions pipeline
- Each test needed a set of assumptions to be met. To make their testing easier, we constructed an assumption pipeline.
      - ANOVA: independence of observations, levene test
      - ANCOVA: independence of pbservations, linearity assumption, normality of residuals,
               homogeneity of slopes, levene test fitted for ANCOVA
     * The assumption of homogoneity of slopes was essential for conducting ANCOVA.
     * If violated, it would mean that there's a significant interaction between the cov and the iv
     * A moderated regression would be conducted instead.
      - Moderated regression: independence of observation, linearity assumption, multicolinearity,          homogeneity of variance that was measured using breusch pagan

## Project Pipeline 
### 1. Exploratory Data Analysis (EDA)
- Data cleaning
- Descriptive statistics for age and brain volume loss.
- Distribution inspection using histograms and boxplots.
- Group comparisons by disease stage and sex.

### 2. ANCOVA 
- To answer: How does **Age** impact brain volume loss among Huntington’s disease patients with somatic expansion in different disease stages?
- Dependent Variable: Brain_Volume_Loss
- Independent Variable: Disease_Stage, Covariate: Age
- Pairwise t-comparrisons as post-hoc

### 3. Moderated Regression 
- Would be conducted if the assumption of homogeneity of slopes was violated
- To answer: How does **Age** impact brain volume loss among Huntington’s  disease patients with somatic expansion?
- Interaction model: Disease_Stage × Age.
- Spotlight (simple slopes) analysis when interaction terms were significant.

### 5. Two-Way ANOVA (Sex × Disease Stage)
- How does **Sex** impact brain volume loss among Huntington’s disease patients with somatic expansion in different disease stages?
- Factors: Sex and Disease_Stage.
- An interactive model is conducted first
- An additive model is conducted second in case the interactive model is insignificant
- Post-hoc analyses (Tukey’s HSD or Games Howell) conducted when appropriate.

  * In each of their pipeline, the validation and assumption pipelines were run
  * Model adjustments were applied when assumptions were violated (e.g., log transformation,          quadratic models)
## Results Summary
To examine whether demographic variables influence brain volume loss, we applied both an ANCOVA model with age as a covariate and a two-way ANOVA testing the joint effects of sex and disease stage.
The ANCOVA showed no significant effect of disease stage on brain volume loss after controlling for age (F(3, 36302) = 0.995, p = 0.392), and age itself was not significantly associated with brain volume loss (F(1, 36302) = 0.659, p = 0.417).
Consistently, the two-way ANOVA revealed no significant main effects of disease stage (F(3, 36299) = 0.435, p = 0.728) or sex (F(1, 36299) = 1.160, p = 0.281), and no significant interaction between sex and disease stage (F(3, 36299) = 0.625, p = 0.599). Removing the interaction term did not change the conclusions, with both disease stage (F(3, 36302) = 0.999, p = 0.392) and sex (F(1, 36302) = 0.021, p = 0.885) remaining non-significant.
Overall, these results indicate that, in this dataset, neither age nor sex significantly explains variation in brain volume loss beyond disease stage.
Although the results were not statistically significant, the analysis was conducted using a rigorous and assumption-aware statistical framework.
## Conclusions
In this project, we tested whether age and sex change (moderate) the relationship between  brain volume loss in patients with Huntington’s disease, while accounting for disease stage.
Using several complementary models (ANCOVA, moderated regression with an Age × Disease Stage interaction, and a two-way ANOVA for Sex × Disease Stage), we did not find significant effects of either age or sex on brain volume loss, and no interaction effects were observed, most likely resulting from synthetic data. Thus, this dataset does not actually reflect the general population of Huntington's disease patients. Therefore, to conduct research that may contribute to our knowledge in regards to the disease, it's best to use a different dataset.
Even though the results were not statistically significant, the analysis followed a strict and transparent workflow, including data validation, outlier handling, assumption testing, and model adjustments when needed.
Overall, the project provides a reusable Python pipeline for analyzing demographic and genetic modifiers in neurodegenerative disease data, and can be directly applied to similar studies on Huntington’s disease or related disorders.

## Instructions for Running the Project

Prerequisites:
- Python 3.8 or higher installed
- Required Python packages listed in requirements.txt

Installation and Execution:
1. Clone the repository:
   git clone https://github.com/carolfarahh/final_project
2. Enter the project folder:
   cd <final_project>
3. Install the required dependencies:
   pip install pandas numpy scipy statsmodels matplotlib seaborn pingouin pytest
4. Run the analysis:
   python main.py

Optional Testing:
- To run unit tests :
  pytest

## References

Ajitkumar, A., Lui, F., & De Jesus, O. (2025). *Huntington disease*. StatPearls Publishing.  
https://www.ncbi.nlm.nih.gov/books/NBK559166/

Donaldson, J., Hensman Moss, D., Ciosi, M., et al. (2026). Huntington disease: Somatic expansion, pathobiology and therapeutics. *Nature Reviews Neurology, 22*, 5–21.  
https://doi.org/10.1038/s41582-025-01159-7

Mohnani, R. (n.d.). *Huntington disease dataset*. Kaggle.  
https://www.kaggle.com/datasets/rajmohnani12/huntington-disease-dataset
