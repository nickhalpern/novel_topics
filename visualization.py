import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import copy
from sklearn.metrics.pairwise import linear_kernel
import cPickle as pickle
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline



def get_data():
    with open('tfidf.dat', 'rb') as infile:
        X = pickle.load(infile)
    X = X.toarray()

    cluster = pd.read_csv('clustering.csv')
    columns = list(cluster.columns)
    columns[0] = 'bookid'
    #columns[-4:] = ['darwin', 'chance_and_luck', 'euclid', 'short_hist_math']
    cluster.columns = columns
    return X, cluster

def graph_clusters(X, cluster, labels, color_map, num_clusters = 5, binary = 0, search_term = 'machine', threshold = 0.1):
    if binary == 1:
        title = search_term
        term = cluster[search_term].values
    else:
        title = 'Book Clusters'

    fig = plt.figure()
    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax = fig.add_subplot(111)

    cluster_range = xrange(0,num_clusters)
    assignment = cluster.cluster_5

    offset = [[0,1.25], [2,1.25], [4,1.25], [1,0], [3,0]]
    label_positions = [[0,2], [2,2.25], [4,2], [1,-.75], [3,-1]]

    for i in cluster_range:
        rows = (assignment.values == i)
        pca = PCA(n_components = 2)
        pca.fit(X[rows,:])
        X_2d = pca.transform(X)
        X_2d[:,0] = X_2d[:,0] + offset[i][0]
        X_2d[:,1] = X_2d[:,1] + offset[i][1]

        if binary == 1:
            X_binary = X_2d[(term > threshold),:]
            X_binary_zero = X_2d[(term <= threshold),:]
            plt.scatter(X_binary_zero[:,0], X_binary_zero[:,1], color = '0.75')
            plt.scatter(X_binary[:,0], X_binary[:,1], color = 'k')
        else:
            plt.scatter(X_2d[:,0], X_2d[:,1], color = color_map[i])
            ax.text(label_positions[i][0], label_positions[i][1], labels[i], ha = 'center')

    plt.axis('off')
    plt.show()
    fig.savefig('graph_clusters.png', facecolor='white', edgecolor='none')

def get_moving_avg(cluster, value = 'year_range', agg = 'count'):
    by_year_all = cluster.pivot_table(index='year', columns='cluster_5', values= value, aggfunc= agg)
    by_year_all = by_year_all.reset_index()
    by_year_all = by_year_all.fillna(0)
    by_year_all['total'] = by_year_all[0] + by_year_all[1] + by_year_all[2] + by_year_all[3] + by_year_all[4]

    year_range = range(1800,1931)

    cluster_0 = []
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []
    cluster_4 = []
    cluster_total = []
    for year_x in year_range:
        if year_x < 1850:
            diff = 10 + 1850 - year_x
        else:
            diff = 10
        ten_year_range = by_year_all[(by_year_all.year > (year_x - diff)) & (by_year_all.year <= year_x)]
        if agg == 'mean':
            past_year_data = ten_year_range[[0, 1, 2, 3, 4, 'total']].mean(axis=0)
        else:
            past_year_data = ten_year_range[[0, 1, 2, 3, 4, 'total']].sum(axis=0)
        cluster_0.append(past_year_data[0])
        cluster_1.append(past_year_data[1])
        cluster_2.append(past_year_data[2])
        cluster_3.append(past_year_data[3])
        cluster_4.append(past_year_data[4])
        cluster_total.append(past_year_data.total)

    moving_avg_df = pd.DataFrame({'year': year_range, 'cluster_0': cluster_0, 'cluster_1': cluster_1, 'cluster_2': cluster_2, 'cluster_3': cluster_3, 'cluster_4': cluster_4, 'cluster_total': cluster_total})
    return moving_avg_df

def plot_by_cluster(moving_avg_df, labels, color_map, agg_type ='sum', filename_out = 'cluster_over_time.png', title = 'Topic Cluster Over Time'):

    fig = plt.figure(figsize = (20,12))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax = fig.add_subplot(111, axisbg = 'white')

    cluster_range = xrange(0,5)

    for x in xrange(0,5):
        column_name = 'cluster_{}'.format(x)

        if agg_type == 'sum':
            spl = UnivariateSpline(moving_avg_df.year, moving_avg_df[column_name]/moving_avg_df['cluster_total'])
        else:
            spl = UnivariateSpline(moving_avg_df.year, moving_avg_df[column_name])
        xs = np.linspace(1800,1923, 1000)
        spl.set_smoothing_factor(.05)
        plt.plot(xs, np.clip(spl(xs),0,1), label = labels[x], color = color_map[x], linewidth = 2)

    plt.axis([1800,1923,0,1])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])

    plt.legend(labels, fontsize = 'small', frameon = False, loc="best")

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.savefig(filename_out, facecolor='white', edgecolor='none')

if __name__ == '__main__':
    X, cluster = get_data()
    labels = ['cowboys and detectives', 'castles and thrones', 'ships, islands and beaches', 'gritty reality', 'drawingrooms']
    color_map = ['#CD5C5C', '#00FF00', '#3498DB', '#DCEF28', '#EB984E']

    ### get the graph of five clusters
    #graph_clusters(X, cluster, labels, color_map)

    ### graph frequency of cluster over time
    #df_cluster_by_year = get_moving_avg(cluster)
    #plot_by_cluster(df_cluster_by_year, labels, color_map, agg_type = 'sum', filename_out = 'cluster_freq_over_time.png')

    ### graph frequency of cluster over time
    cluster_american = cluster[cluster.nationality == 'American']
    df_cluster_by_year = get_moving_avg(cluster_american)
    plot_by_cluster(df_cluster_by_year, labels, color_map, agg_type = 'sum', filename_out = 'cluster_freq_over_time_us.png', title = 'American Authors')

    ### graph frequency of cluster over time
    cluster_british = cluster[cluster.nationality == 'British']
    df_cluster_by_year = get_moving_avg(cluster_british)
    plot_by_cluster(df_cluster_by_year, labels, color_map, agg_type = 'sum', filename_out = 'cluster_freq_over_time_uk.png', title = 'British Authors')



    ### graph frequency of cluster over time
    #df_sentiment_by_year = get_moving_avg(cluster, value = 'pos_score', agg = 'mean')
    #plot_by_cluster(df_sentiment_by_year, labels, color_map, agg_type = 'mean', filename_out = 'pos_sentiment.png', title = 'Book Sentiment Over Time')

    # farmer, farmers, wheat, plant (ranch) (wilderness)
    # machinery, chimney, manager, engine, electric (poverty)
