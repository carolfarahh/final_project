# Demographic Moderation of Brain Volume Loss in Huntington’s Disease  
### A Statistical Analysis of Somatic Expansion Effects
## Authors
- Amane Egbaria
- Adi Drawshey
- Carol Farah

## Project Description
This project investigates how **demographic variables (age and sex)** moderate **brain volume loss** among patients with **Huntington’s disease (HD)** in the context of **somatic CAG repeat expansion**.  
Huntington’s disease is a genetic neurodegenerative disorder caused by an expansion of CAG repeats in the *HTT* gene, leading to progressive motor, cognitive, and structural brain deterioration.

The analysis applies a **rigorous statistical framework** combining exploratory data analysis, regression-based modeling, and factorial designs to assess whether age or sex significantly influence brain volume loss beyond disease stage and genetic instability.
## Research Question
**How do demographics (age and sex) impact brain volume loss among Huntington’s disease patients with somatic expansion?**
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

**Genetic / Biological**
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
│   ├── data_cleaning.py
│   ├── data_import.py
│   ├── data_visualization.py
│   └── statistical_analysis.py
└── tests/
    ├── test_data_analysis.py
    ├── test_data_cleaning.py
    ├── test_data_import.py
    └── test_data_visualization.py

```
## Analysis Pipeline (Key Stages)

### 1. Exploratory Data Analysis (EDA)
- Descriptive statistics for age and brain volume loss.
- Distribution inspection using histograms and boxplots.
- Group comparisons by disease stage and sex.
- Scatter plots examining the relationship between age and brain volume loss.

### 2. Correlation Analysis
- Pearson correlation between age and brain volume loss.
- Visual assessment using scatter plots with best-fit regression lines.
- Linearity checks prior to covariate-based modeling.

### 3. ANCOVA (Age as Covariate)
- Dependent Variable: Brain_Volume_Loss  
- Independent Variable: Disease_Stage  
- Covariate: Age  

Assumptions tested:
- Linearity
- Homogeneity of regression slopes
- Normality of residuals
- Homogeneity of variance

Model adjustments were applied when assumptions were violated (e.g., log transformation, quadratic age term).

### 4. Moderated Regression
- Interaction model: Disease_Stage × Age.
- Heteroskedasticity-consistent (HC) standard errors.
- Spotlight (simple slopes) analysis when interaction terms were significant.

### 5. Two-Way ANOVA (Sex × Disease Stage)
- Factors: Sex and Disease_Stage.
- Testing for main effects and interaction effects.
- Post-hoc analyses (Tukey’s HSD) conducted when appropriate.
## Results Summary
- No statistically significant effects of **age** on brain volume loss were found after controlling for disease stage and somatic expansion.
- No statistically significant effects of **sex** on brain volume loss were detected.
- No significant interaction effects were observed in the ANCOVA or moderated regression analyses.
- Two-way ANOVA revealed no significant main effects or interaction between sex and disease stage.

Although the results were not statistically significant, the analysis was conducted using a rigorous and assumption-aware statistical framework.
## Conclusions
- Demographic variables (age and sex) did not significantly moderate brain volume loss among Huntington’s disease patients with somatic expansion.
- Brain volume loss appears to be driven primarily by disease-specific biological mechanisms rather than demographic factors.
- The study highlights the importance of rigorous statistical testing and assumption checking, even when results are null.
- This analytical framework can be applied to future neurogenetic studies investigating disease modifiers.
## Instructions for Running the Project

Prerequisites:
- Python 3.8 or higher installed
- Required Python packages listed in requirements.txt

Installation and Execution:
1. Clone the repository:
   git clone https://github.com/<USERNAME>/<REPOSITORY_NAME>.git
2. Enter the project folder:
   cd <REPOSITORY_NAME>
3. Install the required dependencies:
   pip install -r requirements.txt
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
