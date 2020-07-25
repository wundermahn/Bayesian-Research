# Necessary imports
import pandas as pd, numpy as np, plotly, plotly.express as px, warnings, gc, plotly.offline as plotly_offline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

# Create a function to automate this
def optimize_k_means(data, verbose):

    # Set variables for the best k, and the best silhouette coefficient
    best_k = 0
    best_sil_coe = 0
    
    # Lists of K values and their scores
    k = []
    score = []
    
    # Loop through 2-8 to check # of clusters
    for i in range(2,8):
        # Create a K Means algorithm
        clusterer = KMeans(n_clusters=i, random_state=np.random.randint(1,100))
        # Fit the algorithm to the data
        cluster_labels = clusterer.fit_predict(data)
        # Find the average silhouette coefficient
        silhouette_avg = silhouette_score(data, cluster_labels)
        
        # If verbose is enabled, print out the average
        if verbose:
            print("For n_clusters =", i,
                "The average silhouette_score is :", silhouette_avg) 

        # If the current is better than the best
        if silhouette_avg > best_sil_coe:
            best_k = i
            best_sil_coe = silhouette_avg
        
        # Append the k and its score
        k.append(i)
        score.append(silhouette_avg)
    
    # Create a dataframe of K values and scores
    df = pd.DataFrame(list(zip(k, score)), columns=['K Value', 'Sil. Coe'])
    
    # Return the best k and silhouette coefficient, and the overall dataframe
    return best_k, best_sil_coe, df

# Function to remove outliers from plotting data
def remove_outliers(df, col, min_quant, max_quant):
    # Lower quantile
    Qx = df[col].quantile(min_quant)
    # Upper quantile
    Qy = df[col].quantile(max_quant)
    # Interquantile range
    IQR = Qy - Qx

    # Return everything in between the IQR
    return df.query('(@Qx - 1.5 * @IQR) <= '+col+' <= (@Qy + 1.5 * @IQR)')

def plot_kmeans(dimensions, k, data, classes):
    # Assure we are plotting in 2D or 3D only
    assert dimensions in [2,3]
    
    # Create a PCA transformer with the # of specified components
    pca = PCA(n_components=dimensions)
    
    # Fit that against the raw data
    X = pca.fit_transform(data)
    
    if(dimensions == 3):
        # Create a dataframe containing the PCAs
        dataset = pd.DataFrame({'PCA1': X[:, 0], 'PCA2': X[:, 1], 'PCA3': X[:, 2]})
        
        # Create KMeans Algo
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=500, random_state=np.random.randint(1,100))
        # Determine which cluster each point belongs to
        dataset['Cluster'] = kmeans.fit_predict(dataset)
        # Bring distribution of clusters
        print(dataset['Cluster'].value_counts())        
        
        # Append the classes
        dataset['class'] = classes
        # Create the figure
        fig = px.scatter_3d(dataset, x='PCA1', y='PCA2', z='PCA3', color='class', 
                        symbol='Cluster', title="3D K Means Sentiment Class Plot")
        # Show the figure
        fig.show() 
    
    # Otherwise, 2D
    else:
        # Create a dataframe containing the PCAs
        dataset = pd.DataFrame({'PCA1': X[:, 0], 'PCA2': X[:, 1]})
        
        # Create KMeans Algo
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=500, random_state=np.random.randint(1,100))
        # Determine which cluster each point belongs to
        dataset['Cluster'] = kmeans.fit_predict(dataset)
        # Bring distribution of clusters
        print(dataset['Cluster'].value_counts())   
        
        # Append the classes
        dataset['class'] = classes
        # Create the figure
        fig = px.scatter(dataset, x='PCA1', y='PCA2', color='class', 
                        symbol='Cluster', title="2D K Means Sentiment Class Plot")
        # Show the figure
        fig.show()     
        
def plot_data(dimensions, data, classes):
    # Assure we are plotting in 2D or 3D only
    assert dimensions in [2,3]
    
    # Create a PCA transformer with the # of specified components
    pca = PCA(n_components=dimensions)
    
    # Fit that against the raw data
    X = pca.fit_transform(data)
    
    if(dimensions == 3):
        # Create a dataframe containing the PCAs
        dataset = pd.DataFrame({'PCA1': X[:, 0], 'PCA2': X[:, 1], 'PCA3': X[:, 2]})
        # Append the classes
        dataset['class'] = classes
        # Create the figure
        fig = px.scatter_3d(dataset, x='PCA1', y='PCA2', z='PCA3', color='class', 
                        title="3D Sentiment Class Plot")
        # Show the figure
        fig.show() 
    
    # Otherwise, 2D
    else:
        # Create a dataframe containing the PCAs
        dataset = pd.DataFrame({'PCA1': X[:, 0], 'PCA2': X[:, 1]})
        # Append the classes
        dataset['class'] = classes
        # Create the figure
        fig = px.scatter(dataset, x='PCA1', y='PCA2', color='class', 
                        title="2D Sentiment Class Plot")
        # Show the figure
        fig.show()         