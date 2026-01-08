from src.statistical_analysis import gameshowell, one_way_anova
import warnings
from scipy.stats._stats_py import DegenerateDataWarning
import numpy as np
import scipy.stats


#The same test can be used for the other post-hoc test, because they have the same concept
def games_howell_test(df, dependent_variable, independent_variable): 
    warnings.simplefilter('error', RuntimeWarning) #We had to import the warnings library for this one

    try:
        gameshowell(df, dependent_variable, independent_variable)
    except ValueError: 
        print("Factor doesn't have enough levels")   
    except TypeError:
        print("Values in independent variable column contain strings") 
    except RuntimeWarning:
        print("Variance is equal to zero")
    except Exception as e:
        print("An error occured")
    else:
        print("You may proceed with the analysis")

# def anova_test(df, dependent_variable, factor):
#     try :
#         one_way_anova(df, dependent_variable, factor)
#     except DegenerateDataWarning:
#         print("Observations number is smaller or equal to number of levels")
#     else:
#         print("You may proceed")

    


