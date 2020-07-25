# Necessary imports
import nltk, re, pandas as pd, xml, os, sklearn, string, numpy as np, time, pickle, gc as gc, warnings
import seaborn as sns, matplotlib.pyplot as plt, plotly.express as px, plotly.offline as plotly_offline
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedShuffleSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from nltk.stem.snowball import SnowballStemmer

# Data Scaling / Normalization
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler

# Custom Library
from research_utils import *
from statistics_utils import *

# Normality Test Imports
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from scipy.stats import shapiro, normaltest, anderson

# Other settings
gc.enable()
warnings.filterwarnings("ignore")
plotly_offline.init_notebook_mode(connected = True)


# Function to plot qq plots for every column in a dataframe
def df_qq_plot(data):
    for col in data.columns:
        qqplot(data[col].to_numpy(), line='s')
        pyplot.show()

# Function to determine % of columns in dataset that are potentially normally distributed
# According to Shapiro-Wilk
def shapiro_wilk(data, alpha, avg):
    # Counts for the # of columns that reject the null hypothesis
    count_true = 0
    count_false = 0
    
    # If you want averages returned, create lists for that
    if avg:
        ps = []
        stats = []
    
    # Loop through all the columns
    for col in data.columns:
        # Run the shapiro test
        stat, p = shapiro(data[col].to_numpy())
        p=float(p)
        # If you want the averages
        if avg:
            # Append those
            ps.append(p)
            stats.append(stat)
        
        # If p is greater than the alpha value passed
        if p > alpha:
            # It is true
            count_true += 1
        # Otherwise
        else:
            # It is false
            count_false += 1
    
    # If you want the averages, return the % of Gaussian columns and the averages
    if avg:
        return 100 * (count_true / (count_true+count_false)), float(np.average(np.asarray(ps))), float(np.average(np.asarray(stats)))
    # Otherwise, just return the % of Gaussian columns
    else:
        return 100 * (count_true / (count_true+count_false))

# Function to determine % of columns in dataset that are potentially normally distributed
# According to Dagastino K^2
def dagastino(data, alpha, avg):
    # Counts for the # of columns that reject the null hypothesis
    count_true = 0
    count_false = 0
    
    # If you want averages returned, create lists for that
    if avg:
        ps = []
        stats = []
    
    # Loop through all the columns
    for col in data.columns:
        # Run the shapiro test
        stat, p = normaltest(data[col].to_numpy())
        p=float(p)
        # If you want the averages
        if avg:
            # Append those
            ps.append(p)
            stats.append(stat)
        
        # If p is greater than the alpha value passed
        if p > alpha:
            # It is true
            count_true += 1
        # Otherwise
        else:
            # It is false
            count_false += 1
    
    # If you want the averages, return the % of Gaussian columns and the averages
    if avg:
        return 100 * (count_true / (count_true+count_false)), float(np.average(np.asarray(ps))), float(np.average(np.asarray(stats)))
    # Otherwise, just return the % of Gaussian columns
    else:
        return 100 * (count_true / (count_true+count_false))

# Function to determine % of columns in dataset that are potentially normally distributed
# According to the Anderson-Darling Test
def anderson_darling_test(data, avg):
    # Counts for the # of columns that reject the null hypothesis
    count_true = 0
    count_false = 0

    # If you want averages returned, create lists for that
    if avg:
        sig_levels = []
        crit_values = []
    
    # Loop through all the columns
    for col in data.columns:
        # Collect the results from the anderson-darling test
        result = anderson(data[col].to_numpy(), dist='norm')
        # Loop through the critical values
        for i in range(len(result.critical_values)):
            # Grab the significance level and critical value
            sl, cv = result.significance_level[i], result.critical_values[i]
            
            # If you want averages, append them
            if avg:
                sig_levels.append(result.significance_level[i])
                crit_values.append(result.critical_values[i])
            
            # If the stat is less than the critical value, it is normally distributed
            if result.statistic < result.critical_values[i]:
                count_true += 1
            # Otherwise, it is not
            else:
                count_false += 1    
                    
    # If you want averages, return that
    if avg:
        return 100 * (count_true / (count_true+count_false)), np.average(np.asarray(sig_levels)), np.average(np.asarray(crit_values))
    # Otherwise, just return the % of columns
    else:
        return 100 * (count_true / (count_true+count_false))