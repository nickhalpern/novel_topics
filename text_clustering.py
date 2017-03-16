import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
import wikipedia
import copy
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import PCA, TruncatedSVD
import cPickle as pickle
# remember to delete duplicate title
# pride and prejudice, Dorothy and the Wizard in Oz

'''
FOR TRAINING
Agriculture
    A Short History of English Agriculture
    Essays in Natural History and Agriculture
    Agriculture for Beginners, Revised Edition
    Agriculture in Virginia, 1607-1699
    The Elements of Agriculture, A Book for Young Farmers, with Questions Prepared for the Use of Schools
    Notes on Agriculture in Cyprus and Its Products
    What I know of farming:, a series of brief and plain expositions of practical, agriculture as an art based upon science
    Makers of Modern Agriculture
    Observations on the Effects of the Corn Laws, and of a Rise or Fall in the Price of Corn on the Agriculture and General Wealth of the Country
    Atoms in Agriculture, (Revised)
Industrial Revolution
    (Hard Times)
    (North and South)
    Textiles, For Commercial, Industrial, and Domestic Arts Schools; Also Adapted to Those Engaged in Wholesale and Retail Dry Goods, Wool, Cotton, and Dressmaker's Trades
    Industrial Biography, Iron Workers and Tool Makers
    Industrial Progress and Human Economics
    An Introduction to the Industrial and Social History of England
    Life in a Railway Factory
'''


def get_wiki(term):
    w_page = wikipedia.page(term)
    return w_page.content


def get_words(filename):
    doc = open(filename)
    words = []
    for line in doc:
        line = line.translate(None, string.punctuation)
        try:
            line = line.decode('utf-8').strip()
            line_words = word_tokenize(line)
            for word in line_words:
                if word == word.lower():
                    if word in stop:
                        pass
                    elif len(word) < 5: ######
                        pass
                    else:
                        words.append(word)
        except:
            pass
    return ' '.join(words)


def vectorize(mat, max_df=0.5,use_idf=True) :
    vectorizer = TfidfVectorizer(max_df=max_df, max_features = 1000,
                                stop_words='english', use_idf=use_idf)
    return vectorizer.fit_transform(mat), vectorizer


def vectorize_n_kmeans(vectorized_data, vectorizer, n = 14):
    #vectorized_data, vectorizer = vectorize(mat)


    kmeans = KMeans(n_clusters=n).fit(vectorized_data)

    cluster_assignment = kmeans.predict(vectorized_data)

    centroids = kmeans.cluster_centers_
    # first_10 = lambda x: sorted(x, reverse=True)[:10]

    thres = lambda x:  sorted(x[x > 0.05], reverse=True)

    cluster_words = vectorizer.inverse_transform(centroids)

    y = centroids * (centroids > 0.05)
    y2 = centroids * (centroids > 0.03) * (centroids < 0.05)

    top_words = vectorizer.inverse_transform(y)
    secondardy_words = vectorizer.inverse_transform(y2)

    return vectorized_data, kmeans, top_words, secondardy_words, cluster_assignment


def plot_2d(X, cluster_assignment):
    cluster_range = xrange(0,max(cluster_assignment) + 1)

    for i in cluster_range:
        rows = (assignment.values == i)
        color_map = ['b', 'g', 'r', 'c', 'y']
        offset = [[1,1], [2.25,1], [0.5, -0.25], [1.5,0], [0,1.5]]
        pca = PCA(n_components = 2)
        pca.fit(X[rows,:])
        X_2d = pca.transform(X)
        X_2d[:,0] = X_2d[:,0] + offset[i][0]
        X_2d[:,1] = X_2d[:,1] + offset[i][1]
        plt.scatter(X_2d[:,0], X_2d[:,1], color = color_map[i])

    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    df = pd.read_csv('allbookdata.csv')
    stop = set(stopwords.words('english'))
    snowball = SnowballStemmer('english')

    training_terms = ['farm', 'war', 'machine', 'family', 'philosophy', 'world war one', \
            'us civil war', 'art', 'sailor', 'feudalism', 'christianity', 'literature', \
            'science', 'factory', 'statistics', 'Race (human categorization)', 'probability', 'industry', 'novel', \
            'city', 'fiction', 'nation', 'nature', 'revolution', 'liberalism', 'evolution', \
            'sex', 'taste', 'Feelings', 'manual labor', 'democracy', 'poetry', 'Charity (practice)', 'capitalism', \
            'country']

    other_pg_texts = ['ebooks-unzipped/1228.txt', 'ebooks-unzipped/chance_and_luck.tex', 'ebooks-unzipped/euclid.tex', 'ebooks-unzipped/short_hist_math.tex']
    other_texts = [get_words(x) for x in other_pg_texts]

    wiki_documents = [get_wiki(t) for t in training_terms]

    novels = [get_words(df.file[i]) for i in xrange(len(df))]



    documents = copy.copy(wiki_documents)
    for other in other_texts:
        documents.append(other)
    for novel in novels:
        documents.append(novel)

    vectorized_data, vectorizer = vectorize(novels)

    with open('tfidf.dat', 'wb') as outfile:
        pickle.dump(vectorized_data, outfile, pickle.HIGHEST_PROTOCOL)

    top_words_dict = {}
    score = []
    cluster_assignment_dict = {}

    k_chosen_range = xrange(5,6) #xrange(3,20)
    for k_chosen in k_chosen_range:
        tfidfed, kmeans, top_words, secondardy_words, cluster_assignment = vectorize_n_kmeans(vectorized_data, vectorizer, n = k_chosen)
        top_words_dict[k_chosen] = top_words
        cluster_assignment_dict[k_chosen] = cluster_assignment
        score.append(kmeans.inertia_)
        print 'Completed Cluster {}:     Score: {}'.format(k_chosen, kmeans.inertia_)
    #####
    #score_df = pd.DataFrame({'number_of_clusters': list(k_chosen_range), 'score': score})
    #score_df.to_csv('score.csv')

    cluster_num = []
    cluster_id = []
    word_array = []
    for key in top_words_dict:
        for j in xrange(len(top_words_dict[key])):
            cluster_num.append(key)
            cluster_id.append(j)
            word_array.append(', '.join(top_words_dict[key][j]))

    word_df = pd.DataFrame({'number_of_clusters': cluster_num, 'cluster_id': cluster_id, 'top_words': word_array})
    #####
    #word_df.to_csv('top_words.csv', encoding='utf-8')


    all_topics = vectorize(documents)[0]
    cosine_similarities = linear_kernel(all_topics, all_topics)

    all_terms = training_terms + other_pg_texts
    train_length = len(all_terms)
    training_category = []
    category_strength = []
    for i in xrange(len(df)):
        most_sim = np.argmax(cosine_similarities[i+train_length,0:train_length])
        max_sim = np.max(cosine_similarities[i+train_length,0:train_length])
        training_category.append(all_terms[most_sim])
        category_strength.append(max_sim)

    output_df = pd.DataFrame({'book_title': df.title, 'author': df.author, 'year': df.year, 'year_range': df.year_range, 'nationality': df.nationality, 'subject': training_category, 'strength': category_strength})
    for k_chosen in k_chosen_range:
        column_name = 'cluster_{}'.format(k_chosen)
        output_df[column_name] = cluster_assignment_dict[k_chosen]

    for i in xrange(len(all_terms)):
        output_df[all_terms[i]] = cosine_similarities[train_length:, i]

    ######
    #output_df.to_csv('clustering.csv', encoding='utf-8')

    '''
    dense = vectorized_data.todense()
    square_distance = squareform(pdist(dense))

    dendo = dendrogram(linkage(square_distance), labels = df.title)
    plt.show()
    '''


    '''

    books_to_choose = ['Bleak House', 'Hard Times', 'Pride and Prejudice', \
        "The Adventures of Huckleberry Finn, Tom Sawyer's Comrade", \
        'The History of Tom Jones, a Foundling', 'The Sea-Wolf', \
        "Tess of the d'Urbervilles, A Pure Woman", 'The Moonstone', \
        'The Vicar of Bullhampton', 'The Life and Adventures of Robinson Crusoe (1808)', \
        'The Age of Innocence', 'The Beautiful and the Damned', \
        'The Last of the Mohicans, A Narrative of 1757', \
        "Gulliver's Travels, Into Several Remote Regions of the World", \
        'Dorothy and the Wizard in Oz', 'Love Among the Chickens', \
        'Pamela, or Virtue Rewarded']

    df = df[df.title.isin(books_to_choose)].reset_index()
    '''


    '''
    X = vectorized_data.todense()
    pca = PCA(n_components = 2)
    pca.fit(X)
    #print(pca.explained_variance_ratio_)
    X_2d = pca.transform(X)


    '''
