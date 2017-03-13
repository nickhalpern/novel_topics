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
                    else:
                        words.append(word)
        except:
            pass
    return ' '.join(words)


def vectorize(mat, max_df=0.5,use_idf=True) :
    vectorizer = TfidfVectorizer(max_df=max_df, max_features = 1000,
                                stop_words='english', use_idf=use_idf)
    return vectorizer.fit_transform(mat), vectorizer


def vectorize_n_kmeans(mat, n = 14):
    vectorized_data, vectorizer = vectorize(mat)


    kmeans = KMeans(n_clusters=n).fit(vectorized_data)

    cluster_assignment = kmeans.predict(vectorized_data)

    centroids = kmeans.cluster_centers_
    # first_10 = lambda x: sorted(x, reverse=True)[:10]

    thres = lambda x:  sorted(x[x > 0.05], reverse=True)

    cluster_words = vectorizer.inverse_transform(centroids)
    y = centroids * (centroids > 0.05)

    top_words = vectorizer.inverse_transform(y)

    return vectorized_data, kmeans, top_words, cluster_assignment



if __name__ == '__main__':
    df = pd.read_csv('allbookdata.csv')

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

    subset_df = df[df.title.isin(books_to_choose)].reset_index()


    stop = set(stopwords.words('english'))
    snowball = SnowballStemmer('english')

    training_terms = ['farm', 'war', 'machine', 'family', 'philosophy', 'world war one', 'us civil war'] #religion
    wiki_documents = [get_wiki(t) for t in training_terms]

    #novels = [get_words(subset_df.file[i]) for i in xrange(len(subset_df))]
    novels = [get_words(df.file[i]) for i in xrange(len(df))]

    documents = copy.copy(wiki_documents)
    for novel in novels:
        documents.append(novel)

    #vectorize
    tfidfed, kmeans, top_words, cluster_assignment = vectorize_n_kmeans(novels, n = 10)
    #print top_words

    all_topics = vectorize(documents)[0]
    cosine_similarities = linear_kernel(all_topics, all_topics)

    train_length = len(training_terms)
    training_category = []
    category_strength = []
    for i in xrange(len(novels)):
        most_sim = np.argmax(cosine_similarities[i+train_length,0:train_length])
        max_sim = np.max(cosine_similarities[i+train_length,0:train_length])
        training_category.append(training_terms[most_sim])
        category_strength.append(max_sim)
        #print 'cluster: {}, subject: {}, title: {}'.format(cluster_assignment[i],training_terms[most_sim], books_to_choose[i])

    #output_df = pd.DataFrame({'category': cluster_assignment, 'subject': training_category, 'novel': books_to_choose})
    output_df = pd.DataFrame({'category': cluster_assignment, 'subject': training_category, 'strength': category_strength, 'novel': df.title})
    output_df.to_csv('clustering.csv', encoding='utf-8')
    print output_df[['category', 'subject', 'strength', 'novel']]
    '''
    dense = vectorize(documents)[0].todense()
    square_distance = squareform(pdist(dense))

    dendo = dendrogram(linkage(square_distance), labels = data[100:250, -7])
    plt.show()
    '''
