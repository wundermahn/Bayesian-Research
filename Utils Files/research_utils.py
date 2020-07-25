__author__ = "Anthony (Tony) Kelly"
__copyright__ = "Copyright 2020, Johns Hopkins University Whiting School of Engineering"
__credits__ = ["Tony Kelly", "Marc Johnson", "Tony Johnson"]
__license__ = "GPL"
__version__ = "0.9.1"
__maintainer__ = "Tony Kelly"
__email__ = "kellyt419@gmail.com"
__status__ = "Development"

"""

This is a general utility library for functions used to investigate the correlation between data distribution and performance of Naive Bayes classifiers

"""

# Necessary imports
import nltk, re, pandas as pd, xml, os, sklearn, string, numpy as np, time, pickle, gc as gc, warnings
import seaborn as sns, matplotlib.pyplot as plt, plotly.express as px, plotly.offline as plotly_offline
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedShuffleSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from nltk.stem.snowball import SnowballStemmer

gc.enable()

# Function to parse the xml the Amazon reviews come in
def parse_xml(filepath):    
    # Open the file
    with open(filepath) as file:
        # Read the text
        txt = file.read()
        # Parse the XML for all of the ratings (target)
        ratings = [a.string.strip('\n') for a in BeautifulSoup(txt).find_all('rating')]        
        # Parse the XML for all of the text
        txts = [x.string.strip('\n') for x in BeautifulSoup(txt).find_all('review_text')]
    
    # Return a list of ratings and texts (associated)
    return ratings, txts

# Function to return all folder hierarchies for a given parent
def fast_scandir(dirname):
    # First, get all directories inside the parent directory
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    # Then loop through those, and keep expanding down until there are no more directories
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    # Return total list
    return subfolders

# This function removes numbers from an array
def remove_nums(arr): 
    # Declare a regular expression
    pattern = '[0-9]'  
    # Remove the pattern, which is a number
    arr = [re.sub(pattern, '', i) for i in arr]    
    # Return the array with numbers removed
    return arr

# This function cleans the passed in paragraph and parses it
def get_words(para, stem):   
    # Create a set of stop words
    stop_words = set(stopwords.words('english'))
    # Split it into lower case    
    lower = para.lower().split()
    # Remove punctuation
    no_punctuation = (nopunc.translate(str.maketrans('', '', string.punctuation)) for nopunc in lower)
    # Remove integers
    no_integers = remove_nums(no_punctuation)
    # Remove stop words
    dirty_tokens = (data for data in no_integers if data not in stop_words)
    # Ensure it is not empty
    tokens = [data for data in dirty_tokens if data.strip()]
    # Ensure there is more than 1 character to make up the word
    tokens = [data for data in tokens if len(data) > 1]
       
    if stem == True:
        # Perform stemming
        stemmer = SnowballStemmer('english')
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        return stemmed_tokens
    
    else:
        # Return the tokens
        return tokens  
    
# Function to plot the distributions of each word in the df
def plot_df_distributions(df, name):
    # Loop through the df
    for col in df.columns:
        # Create a temp numpy array
        temp = df[col].to_numpy()
        # Plot the distribution
        n, bins, patches = plt.hist(temp, bins=3)
        # Set the title
        plt.suptitle("Data Type: {} | {} Distribution".format(str(name), str(col)))
        # Show the figure
        plt.show()    
        
# Function to plot the distributions of each word in the df
def plot_pos_distributions(df, cols):
    # Loop through the df
    for col in cols:
        # Create a temp numpy array
        temp = df[col].to_numpy()
        # Plot the distribution
        n, bins, patches = plt.hist(temp, bins=10)
        # Set the title
        plt.suptitle("POS Tag: {} | Distribution".format(str(col)))
        # Show the figure
        plt.show()        

# Function to plot the results of the classifiers
def plot_results(df, name, metric):
    # Create a plotly figure, and plot whichever metric (best, worst, average) is passed
    fig = px.line(df, x='Simulation', y=metric, color='Classifier')
    # Update title/labels
    fig.update_layout(title='{} Data Classifier Performance'.format(name), xaxis_title='Simulation',
                             yaxis_title='Accuracy')
    # Display the figure
    fig.show()     
    
    
# This function parses NLTK returned POS tuples
def parse_tuples(list_of_tuples, verbose):
    
    # Declare POS counts
    cnt_noun = 0
    cnt_adj = 0
    cnt_vb = 0
    cnt_other = 0
    
    # Loop through the returned tuples
    for tpl in list_of_tuples:
        
        # NOTE - If needed, verbose printing is available to
        # check for completeness.
        
        # If the word is a noun, increase the noun count
        if('NN' in tpl[1]):
            cnt_noun += 1
            if(verbose):
                print("Noun: {}".format(tpl))
        # If the word is an adjective, increase the adjective count
        elif('JJ' in tpl[1]):
            cnt_adj += 1
            if(verbose):
                print("Adjective: {}".format(tpl))
        # If the word is a verb, increase the verb count
        elif('VB' in tpl[1] or 'VP' in tpl[1]):
            cnt_vb += 1
            if(verbose):
                print("Verb: {}".format(tpl))
        # If the word isn't one of those 3, increase the other count
        else:
            cnt_other += 1
            if(verbose):
                print("Other: {}".format(tpl))
    
    # Return the counts
    return cnt_noun, cnt_adj, cnt_vb, cnt_other    

def plot_correlations(df, name):
    # Set figure and axis size
    fig, ax = plt.subplots(figsize = (15,12))
    # Create a temp copy of the df
    temp = df.copy()
    # https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    # Create a correlation matrix
    corr_matrix = temp.corr(method='spearman').abs()
    
    # Do some cleaning to remove features that have more than a 95% correlation, or a less trhan 1% correlation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95) or any(upper[column] < 0.01)]
    temp.drop(to_drop, axis=1, inplace=True)
    
    # Now, re calculate the correlation with good columns
    corr = temp.corr(method='spearman')
    
    # Create a heatmap
    _ = sns.heatmap(corr, fmt="f", linewidths=0.25, center=0, cmap='coolwarm', linecolor='black')
    # Set the title
    ax.set_title('{} Correlation Heatmap'.format(name))
    
    # Show the plot, then close it
    plt.show()
    plt.close()

# Function to do the corpus training / 
def create_amazon_dataframes(df, binary, classname):
    
    # If binary class, change all 1 and 2 ratings to 0, and 4 and 5 ratings to 1
    if binary:
        df[classname].replace(1.0, 0, inplace=True)
        df[classname].replace(2.0, 0, inplace=True)
        df[classname].replace(4.0, 1, inplace=True)
        df[classname].replace(5.0, 1, inplace=True)
    
    # Create a corpus from the text of the df
    s = pd.Series(df['Text']).astype('str')
    corpus = s.apply(lambda s: ' '.join(get_words(s, True)))
    
    classes = df[classname]
    
    min_df = int(0.05 * df.shape[0])
    
    # Create vectorizers, and vectors, for the three types of vectors we have described:
    # - Frequency, Boolean, and TFIDF
    # With the conditions we have described
    # - At least 5% of all reviews, no more than 99% of all reviews
    trimmed_boolean_vectorizer = CountVectorizer(strip_accents='unicode',
                                                 min_df=int(min_df), max_df = 0.99, 
                                                 binary=True)
    trimmed_tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode', min_df=int(min_df), max_df = 0.99)
    trimmed_count_vectorizer = CountVectorizer(strip_accents='unicode', min_df=int(min_df), max_df = 0.99)

    # Create the vectors
    trimmed_tfidf = trimmed_tfidf_vectorizer.fit_transform(corpus)
    trimmed_count = trimmed_count_vectorizer.fit_transform(corpus)
    trimmed_boolean = trimmed_boolean_vectorizer.fit_transform(corpus)

    # Now, apply each of these vectorizers to the data and create dataframes
    gc.collect()

    # Create the dataframes
    trimmed_boolean_df = pd.DataFrame(data = trimmed_boolean.todense(), columns = trimmed_boolean_vectorizer.get_feature_names())
    trimmed_tfidf_df = pd.DataFrame(data = trimmed_tfidf.todense(), columns = trimmed_tfidf_vectorizer.get_feature_names())
    trimmed_count_df = pd.DataFrame(data = trimmed_count.todense(), columns = trimmed_count_vectorizer.get_feature_names())
    
    # Free up some memory
    gc.collect()    
    
    # Create list of dfs
    dfs = [trimmed_boolean_df, trimmed_tfidf_df, trimmed_count_df]
    names = ["Boolean", "TFIDF", "Count"]
    
    
    # Make sure that we have not lost any data
    for df in dfs:
        assert(len(classes) == df.shape[0])
        
    # Return it
    return dfs, names, classes

def load_imdb_data():
    # Import the data
    boolean = pd.read_csv('Trimmed_Boolean.csv')
    tfidf = pd.read_csv('Trimmed_TFIDF.csv')
    freq = pd.read_csv('Trimmed_Count.csv')

    # Create lists of the dataframes, and their names
    dfs = [boolean, tfidf, freq]
    names = ['Boolean', 'TFIDF', 'Frequency']

    # Store the classes
    classes = tfidf['classification']

    # Drop weird column pandas creates when writing to csv
    for df in dfs:
        df.drop(['Unnamed: 0', 'classification'], axis=1, inplace=True)
    
    # Return the dataframes, their names, and the classes
    return dfs, names, classes