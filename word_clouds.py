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
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import TruncatedSVD, NMF
from os import path
from wordcloud import WordCloud, STOPWORDS


def get_words(filename):
    stop = set(stopwords.words('english'))
    stop.update('didn', 've', 'll', 'don', 'ain', 'couldn', 'doesn', 'wouldn', 'hadn', 'isn', 'hadnt', 'arent', 'youve', 'whats', 'shouldnt', 'theyre', 'maam', 'theyd')
    tbl = string.maketrans('-', ' ')
    doc = open(filename)
    words = []
    for line in doc:
        line = line.translate(tbl)
        line = line.translate(None, string.punctuation)
        try:
            line = line.decode('utf-8').strip()
            line_words = word_tokenize(line)
            for word in line_words:
                if word == word.lower():
                    if word.strip() not in stop:
                        words.append(word.strip())
        except:
            pass
    return ' '.join(words)


def vectorize(mat, max_df=0.5,use_idf=True) :
    stop = set(stopwords.words('english'))
    stop.update('shes', 'av', '8vo', 'youd', 'fourteen', 'yer', 'la', 'weve', \
            'nay', 'twas', 'sez', 'im', 'hed', 'ha', 'whats', 'er', 'wi', 'didn', \
            've', 'll', 'don', 'ain', 'couldn', 'doesn', 'wouldn', 'hadn', 'isn', \
            'hadnt', 'arent', 'youve', 'whats', 'shouldnt', 'theyre', 'maam', 'theyd')

    vectorizer = TfidfVectorizer(max_df=max_df, max_features = 1000,
                                stop_words= stop, use_idf=use_idf)

    vectorized_data = vectorizer.fit_transform(mat)

    with open('tfidf.dat', 'wb') as outfile:
        pickle.dump(vectorized_data, outfile, pickle.HIGHEST_PROTOCOL)

    with open('vocab.dat', 'wb') as outfile:
        pickle.dump(vectorizer.vocabulary_, outfile, pickle.HIGHEST_PROTOCOL)

    pd.DataFrame({'word_list': vectorizer.vocabulary_.keys()}).to_csv('list_of_words.csv')

    return vectorized_data, vectorizer

def reconst_mse(target, left, right):
    return (np.array(target - left.dot(right))**2).mean()

def describe_nmf_results(document_term_mat, W, H, component):
    text_list = []
    stopwords = set(STOPWORDS)
    d = path.dirname(__file__)
    for topic_num, topic in enumerate(H):
        print 'Running Cluster: {} for N Components: {}'.format(topic_num, component)
        text = " ".join([feature_words[i] for i in topic.argsort()[:-21:-1]]
        text_list.append(text)
        wc = WordCloud(background_color="white", max_words=2000, stopwords=stopwords)
        # generate word cloud
        wc.generate(text)
        # store to file
        file_name = 'word_clouds/{}_cluster_{}.png'.format(component, topic_num)
        wc.to_file(path.join(d, file_name))
    return reconst_mse(document_term_mat, W, H)

def get_terms_score(terms, vectorizer):
    idx = []
    for word in terms:
        try:
            fact_index.append(vectorizer.vocabulary_[word])
        except:
            continue
    return np.sum(X[:,idx],axis = 1)

if __name__ == '__main__':

    #### read in the novel data
    df = pd.read_csv('allbookdata.csv')
    novels = [get_words(df.file[i]) for i in xrange(len(df))]
    ####

    #### tfidf the data
    vectorized_data, vectorizer = vectorize(novels)
    ####

    error = []
    components = range(3,10)
    for component in components:
        #### nmf
        print("\n\n---------\nsklearn decomposition")
        feature_words = vectorizer.get_feature_names()
        nmf = NMF(n_components=component)
        W_sklearn = nmf.fit_transform(vectorized_data)
        H_sklearn = nmf.components_
        error_reconst = describe_nmf_results(vectorized_data, W_sklearn, H_sklearn, component)
        error.append(error_reconst)
        ####

    pd.DataFrame({'num_components': components, 'reconst_error': error}).to_csv('nmf_error.csv')

    #### farming vs factory vs slavery
    get_terms_score(['farmer', 'farm', 'plant', 'wheat', 'farmers'], vectorized_data.toarray())

    fact_keywords = ['machinery, chimney, manager, engine, electric']
    fact_index = []
    for word in fact_keywords:
        try:
            fact_index.append(vectorizer.vocabulary_[word])
        except:
            continue
    fact_score = np.sum(X[:,fact_index],axis = 1)
    ####

    # bet, probable, model, science, average, expectation, scientific
