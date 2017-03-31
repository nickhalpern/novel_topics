import matplotlib.patches as mpatches
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
from text_clustering import run_word_cloud
from sklearn import manifold

def get_data():
    with open('tfidf.dat', 'rb') as infile:
        X = pickle.load(infile)
    X = X.toarray()

    with open('vectorizer.dat', 'rb') as infile:
        vectorizer = pickle.load(infile)

    cluster = pd.read_csv('clustering.csv')
    columns = list(cluster.columns)
    columns[0] = 'bookid'
    #columns[-4:] = ['darwin', 'chance_and_luck', 'euclid', 'short_hist_math']
    cluster.columns = columns
    return X, vectorizer, cluster

def graph_clusters(X, cluster, labels, color_map, num_clusters = 5, binary = 0, sent = 0, pos = 0, search_term = 'machine', threshold = 0.1, title = 'Book Clusters', filename = 'graph_clusters.png'):
    if binary == 1:
        title = search_term
        term = cluster[search_term].values
    elif sent == 1:
        category = cluster.category.values
    else:
        title = 'Book Clusters'

    if sent == 1:
        fig = plt.figure(figsize=(6,6))
    else:
        fig = plt.figure(figsize=(10,8))

    fig.suptitle(title, fontsize=16, fontweight='bold')

    ax = fig.add_subplot(111)

    cluster_range = xrange(0,num_clusters)
    assignment = cluster.cluster_5

    offset = [[0,1.25], [2,1.25], [4,1.25], [1,0], [3,0]]
    label_positions = [[0,2.25], [2,2], [4,2], [1,-.7], [3,-0.8]]

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
        elif sent == 1:
            X_pos = X_2d[(category == 1),:]
            X_neg = X_2d[(category == -1),:]
            X_zero = X_2d[(category == 0),:]
            plt.scatter(X_zero[:,0], X_zero[:,1], color = '#d3d3d3')
            if pos == 1:
                plt.scatter(X_pos[:,0], X_pos[:,1], color = '#3498DB', alpha = 0.75)
            if pos == 0:
                plt.scatter(X_neg[:,0], X_neg[:,1], color = '#CD5C5C', alpha = 0.75)
            ax.text(label_positions[i][0], label_positions[i][1], labels[i], ha = 'center', fontsize = 14)
        else:
            plt.scatter(X_2d[:,0], X_2d[:,1], color = color_map[i], s = 20)
            ax.text(label_positions[i][0], label_positions[i][1], labels[i], ha = 'center', fontsize = 14)

    plt.axis('off')
    fig.savefig(filename, facecolor='white', edgecolor='none')

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

def plot_by_cluster(moving_avg_df, labels, color_map, agg_type ='sum', filename_out = 'cluster_over_time.png', title = 'Topic Cluster Over Time', wide = 1):

    if wide == 1:
        fig = plt.figure(figsize = (10,6))
    else:
        fig = plt.figure(figsize = (6,6))

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

def plot_by_cluster2(cluster, labels, color_map, agg_type ='sum', filename_out = 'cluster_over_time.png', title = 'Topic Cluster Over Time', wide = 1):

    book_by_year = pd.DataFrame(cluster.groupby('year').count()['author']).reset_index()
    book_by_year.columns = ['year', 'total']

    if wide == 1:
        fig = plt.figure(figsize = (10,6))
    else:
        fig = plt.figure(figsize = (6,6))

    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax = fig.add_subplot(111, axisbg = 'white')

    cluster_range = xrange(0,5)

    for x in xrange(0,5):

        cluster_by_year = pd.DataFrame(cluster[cluster.cluster_5 == x].groupby('year').count()['author']).reset_index()
        cluster_by_year.columns = ['year', 'cluster_num']
        ww = pd.merge(book_by_year, cluster_by_year, on = 'year', how = 'left')
        ww = ww.fillna(0)
        ww['pct'] = ww.cluster_num / ww.total

        if agg_type == 'sum':
            spl = UnivariateSpline(ww.year, ww.pct)
        else:
            spl = UnivariateSpline(ww.year, ww.cluster_num)
        xs = np.linspace(1800,1923, 1000)
        spl.set_smoothing_factor(10)
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



def plot_by_cluster_single(cluster, moving_avg_df, labels, color_map, agg_type ='sum', filename_out = 'cluster_over_time.png', title = 'Topic Cluster Over Time', wide = 1, cluster_num = 2):

    book_by_year = pd.DataFrame(cluster.groupby('year').count()['author']).reset_index()
    book_by_year.columns = ['year', 'total']
    cluster3_by_year = pd.DataFrame(cluster[cluster.cluster_5 == 2].groupby('year').count()['author']).reset_index()
    cluster3_by_year.columns = ['year', 'cluster_num']
    ww = pd.merge(book_by_year, cluster3_by_year, on = 'year', how = 'left')
    ww = ww.fillna(0)
    ww['pct'] = ww.cluster_num / ww.total

    if wide == 1:
        fig = plt.figure(figsize = (10,6))
    else:
        fig = plt.figure(figsize = (6,6))

    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax = fig.add_subplot(111, axisbg = 'white')

    cluster_range = xrange(0,5)

    column_name = 'cluster_{}'.format(cluster_num)

    #spl = UnivariateSpline(ww.year, ww.pct)
    spl = UnivariateSpline(moving_avg_df.year, moving_avg_df.cluster_2/moving_avg_df.cluster_total)
    xs = np.linspace(1800,1923, 1000)
    spl.set_smoothing_factor(.01)
    plt.plot(xs, np.clip(spl(xs),0,1), label = labels[cluster_num], color = color_map[cluster_num], linewidth = 2)

    plt.scatter(ww.year, ww.pct, s = 5, color = color_map[cluster_num])

    plt.axis([1800,1923,0,0.6])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.annotate('Lewis and Clark Expedition', xy=(1803,0), xycoords='data',
                xytext=(1803,0.4), textcoords='data', rotation = 90,
                size=12, va="center", ha="left", arrowprops=dict(arrowstyle='fancy', facecolor = 'black'))
    '''
    ax.annotate('Indian Removal Act Becomes Law', xy=(1830,0), xycoords='data',
                xytext=(1830,0.4), textcoords='data', rotation = 90,
                size=12, va="center", ha="left", arrowprops=dict(arrowstyle='fancy', facecolor = 'black'))

    ax.annotate('Start of Mexican-American War', xy=(1846,0), xycoords='data',
                xytext=(1846,0.4), textcoords='data', rotation = 90,
                size=12, va="center", ha="left", arrowprops=dict(arrowstyle='fancy', facecolor = 'black'))
    '''
    ax.annotate('Homestead Act Encourages Settlers', xy=(1862,0), xycoords='data',
                xytext=(1862,0.4), textcoords='data', rotation = 90,
                size=12, va="center", ha="left", arrowprops=dict(arrowstyle='fancy', facecolor = 'black'))

    ax.annotate('US Census: the Frontier Has Disappeared', xy=(1890,0), xycoords='data',
                xytext=(1890,0.4), textcoords='data', rotation = 90,
                size=12, va="center", ha="left", arrowprops=dict(arrowstyle='fancy', facecolor = 'black'))


    fig.savefig(filename_out, facecolor='white', edgecolor='none')

def plot_by_cluster_single_dr(cluster, moving_avg_df, labels, color_map, agg_type ='sum', filename_out = 'cluster_over_time.png', title = 'Topic Cluster Over Time', wide = 1, cluster_num = 2):

    book_by_year = pd.DataFrame(cluster.groupby('year').count()['author']).reset_index()
    book_by_year.columns = ['year', 'total']
    cluster3_by_year = pd.DataFrame(cluster[cluster.cluster_5 == 1].groupby('year').count()['author']).reset_index()
    cluster3_by_year.columns = ['year', 'cluster_num']
    ww = pd.merge(book_by_year, cluster3_by_year, on = 'year', how = 'left')
    ww = ww.fillna(0)
    ww['pct'] = ww.cluster_num / ww.total

    if wide == 1:
        fig = plt.figure(figsize = (10,6))
    else:
        fig = plt.figure(figsize = (6,6))

    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax = fig.add_subplot(111, axisbg = 'white')

    cluster_range = xrange(0,5)

    column_name = 'cluster_{}'.format(cluster_num)

    #spl = UnivariateSpline(ww.year, ww.pct)
    spl = UnivariateSpline(moving_avg_df.year, moving_avg_df[column_name]/moving_avg_df.cluster_total)
    xs = np.linspace(1800,1923, 1000)
    spl.set_smoothing_factor(.05)
    plt.plot(xs, np.clip(spl(xs),0,1), label = '"Drawingroom" Books (% Total Novels)', color = color_map[cluster_num], linewidth = 2)

    occupation_df = pd.read_csv('occupation.csv')
    plt.plot(occupation_df.year, occupation_df.class1_pct, color = 'k', linewidth = 2, linestyle = '--', label = 'Professional Class (% Labor Force)')

    plt.legend(fontsize = 'small', frameon = False, loc="best")

    plt.scatter(ww.year, ww.pct, s = 5, color = color_map[cluster_num])


    plt.axis([1800,1923,0,1.01])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.savefig(filename_out, facecolor='white', edgecolor='none')




def sentiment(cluster):
    cluster['net'] = cluster.pos_score - cluster.neg_score

    smoothing_factor = 0.02
    year_df_m = cluster.groupby('year').mean().reset_index()

    ts = year_df_m[['year','net']]
    #ts = ts[ts.year >= 1850]
    spl = UnivariateSpline(ts.year, ts.net)
    spl.set_smoothing_factor(smoothing_factor)

    xs = np.linspace(1605,2017,413)

    stdev = np.std(cluster.net)

    spline_df = pd.DataFrame({'year': xs, 'spline_mean': spl(xs)})
    spline_df['mean_plus'] = spline_df.spline_mean + stdev
    spline_df['mean_minus'] = spline_df.spline_mean - stdev

    cluster_with_mean = pd.merge(cluster[['year', 'author', 'book_title', 'net']], spline_df, on = 'year')

    above_sd = (cluster_with_mean.net > cluster_with_mean.mean_plus)
    below_sd = (cluster_with_mean.net < cluster_with_mean.mean_minus)
    cluster_with_mean['category'] = 1.*above_sd - 1.*below_sd

    fig = plt.figure()
    fig.suptitle('Net Sentiment', fontsize=12, fontweight='bold')
    ax = fig.add_subplot(111, axisbg = 'white')
    ## all books
    plt.plot(spline_df.year, spline_df.spline_mean, linewidth = 2, c = 'k')
    plt.plot(spline_df.year, spline_df.mean_plus, linewidth = 1, c = 'k', ls = '--')
    plt.plot(spline_df.year, spline_df.mean_minus, linewidth = 1, c = 'k', ls = '--')

    c_pos = cluster_with_mean[cluster_with_mean.category == 1]
    c_neg = cluster_with_mean[cluster_with_mean.category == -1]
    c_neu = cluster_with_mean[cluster_with_mean.category == 0]

    plt.scatter(c_pos.year, c_pos.net, s = 3, color = '#3498DB', edgecolors = None)
    plt.scatter(c_neg.year, c_neg.net, s = 3, color = '#CD5C5C', edgecolors = None)
    plt.scatter(c_neu.year, c_neu.net, s = 3, color = '#d3d3d3', edgecolors = None)

    plt.axis([1850,1923,-0.2,0.3])

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.savefig('sentiment_over_time.png', facecolor='white', edgecolor='none')

    return cluster_with_mean.category

def plot_tsne(X, cluster, labels, color_map):
    tsne = manifold.TSNE(n_components = 2, init = 'pca')
    Y = tsne.fit_transform(X)
    fig = plt.figure(figsize = (6,6))
    fig.suptitle('Book Clusters (TSNE)', fontsize=12, fontweight='bold')
    ax = fig.add_subplot(111, axisbg = 'white')

    positions = [[10,-22], [20,20], [-20,-20], [26,-14], [-30,0]]

    for i in xrange(0,5):
        rows = np.array(cluster.cluster_5 == i, dtype = bool)
        Y_c = Y[rows,:]
        plt.scatter(Y_c[:, 0], Y_c[:, 1], color = color_map[i], cmap=plt.cm.Spectral, s = 25, alpha = 0.75, label = labels[i])
        ax.text(positions[i][0], positions[i][1], labels[i], ha = 'center', fontsize = 11, color = color_map[i], fontweight='bold')

    plt.axis('off')
    fig.savefig('tsne.png', facecolor='white', edgecolor='none')


if __name__ == '__main__':
    X, vectorizer, cluster = get_data()
    labels = ['ships, islands and beaches', 'drawingrooms', 'wild west', 'castles and thrones', 'city intrigue']
    color_map = ['#CD5C5C', '#00FF00', '#3498DB', '#DCEF28', '#EB984E']

    ### plot tsne
    plot_tsne(X, cluster, labels, color_map)

    '''
    ### get the graph of five clusters
    graph_clusters(X, cluster, labels, color_map)


    ### graph frequency of cluster over time
    df_cluster_by_year = get_moving_avg(cluster)
    plot_by_cluster(df_cluster_by_year, labels, color_map, agg_type = 'sum', filename_out = 'cluster_freq_over_time.png')

    ### graph frequency of cluster over time
    cluster_american = cluster[cluster.nationality == 'American']
    df_cluster_by_year = get_moving_avg(cluster_american)
    #plot_by_cluster(df_cluster_by_year, labels, color_map, agg_type = 'sum', filename_out = 'cluster_freq_over_time_us.png', title = 'American Authors', wide = 0)
    #plot_by_cluster2(cluster_american, labels, color_map, agg_type = 'sum', filename_out = 'cluster_freq_over_time_us_2.png', title = 'American Authors', wide = 0)
    plot_by_cluster_single(cluster_american, df_cluster_by_year, labels, color_map, agg_type = 'sum', filename_out = 'cluster_freq_over_time_us_ww.png', title = 'Frequency of "Wild West" Cluster (American Authors)', cluster_num = 2)

    ### graph frequency of cluster over time
    cluster_british = cluster[cluster.nationality == 'British']
    df_cluster_by_year = get_moving_avg(cluster_british)
    #plot_by_cluster(df_cluster_by_year, labels, color_map, agg_type = 'sum', filename_out = 'cluster_freq_over_time_uk.png', title = 'British Authors', wide = 0)
    plot_by_cluster_single_dr(cluster_british, df_cluster_by_year, labels, color_map, agg_type = 'sum', filename_out = 'cluster_freq_over_time_uk_dr.png', title = 'Frequency of "Drawingroom" Cluster (British Authors)', cluster_num = 1)



    category = sentiment(cluster)

    cluster['category'] = category
    graph_clusters(X, cluster, labels, color_map, sent = 1, pos = 0, title = 'Sentiment Analysis: Negative', filename = 'sentiment_clusters_neg.png')
    graph_clusters(X, cluster, labels, color_map, sent = 1, pos = 1, title = 'Sentiment Analysis: Positive', filename = 'sentiment_clusters_pos.png')


    feature_words = vectorizer.get_feature_names()
    word_sentiment = X.T.dot(category)

    pos_word_list = []
    for element in word_sentiment.argsort()[:30]:
        pos_word_list.append(feature_words[element])
    run_word_cloud(' '.join(pos_word_list), '', 'pos_word_list', max_word = 30)

    neg_word_list = []
    for element in word_sentiment.argsort()[:-30:-1]:
        neg_word_list.append(feature_words[element])
    run_word_cloud(' '.join(neg_word_list), '', 'neg_word_list', max_word = 30)
    '''
