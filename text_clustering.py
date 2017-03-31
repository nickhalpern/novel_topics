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
import nltk
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
from wordcloud import WordCloud, STOPWORDS
from os import path
import Markov

# what is most highly correlated with plant through time

def get_wiki(term):
    w_page = wikipedia.page(term)
    return w_page.content


def get_words(filename, all_words = False):
    stop = set(stopwords.words('english'))
    stop.update('shes', 'av', '8vo', 'youd', 'fourteen', 'yer', 'la', 'weve', \
            'nay', 'twas', 'sez', 'im', 'hed', 'ha', 'whats', 'er', 'wi', 'didn', \
            've', 'll', 'don', 'ain', 'couldn', 'doesn', 'wouldn', 'hadn', 'isn', \
            'hadnt', 'arent', 'youve', 'whats', 'shouldnt', 'theyre', 'maam', 'theyd', \
            'th', 'weve', 'fer', 'shes', 'hasnt','hadnt', 'un', 'hed', 'im', 'er', \
            'ah', 'ay', 'wi', 'youd', 'aint')
    #tbl = string.maketrans('-', ' ')
    doc = open(filename)
    words = []
    for line in doc:
        #line = line.translate(tbl)
        line = line.translate(None, string.punctuation)
        try:
            line = line.decode('utf-8').strip()
            line_words = word_tokenize(line)
            for word in line_words:
                if all_words == False:
                    if word == word.lower():
                        if word.strip() not in stop:
                            words.append(word.strip())
                else:
                    words.append(word.lower().strip())
        except:
            pass
    return ' '.join(words)


def vectorize(mat, max_df=0.5,use_idf=True) :
    stop = set(stopwords.words('english'))
    stop.update('shes', 'av', '8vo', 'youd', 'fourteen', 'yer', 'la', 'weve', \
            'nay', 'twas', 'sez', 'im', 'hed', 'ha', 'whats', 'er', 'wi', 'didn', \
            've', 'll', 'don', 'ain', 'couldn', 'doesn', 'wouldn', 'hadn', 'isn', \
            'hadnt', 'arent', 'youve', 'whats', 'shouldnt', 'theyre', 'maam', 'theyd', \
            'th', 'weve', 'fer', 'shes', 'hasnt','hadnt', 'un', 'hed', 'im', 'er', \
            'ah', 'ay', 'wi', 'youd', 'aint')

    '''
    vocab = pd.read_csv('vocab.csv')
    vocab_words = list(vocab.vocab)

    vectorizer = TfidfVectorizer(max_df=max_df, max_features = 3000,
                                stop_words= stop, use_idf=use_idf, vocabulary = vocab_words)
    '''
    vectorizer = TfidfVectorizer(max_df=max_df, max_features = 3000,
                                stop_words= stop, use_idf=use_idf)

    vectorized_data = vectorizer.fit_transform(mat)

    with open('tfidf.dat', 'wb') as outfile:
        pickle.dump(vectorized_data, outfile, pickle.HIGHEST_PROTOCOL)

    with open('vocab.dat', 'wb') as outfile:
        pickle.dump(vectorizer.vocabulary_, outfile, pickle.HIGHEST_PROTOCOL)

    pd.DataFrame({'word_list': vectorizer.vocabulary_.keys()}).to_csv('list_of_words.csv')

    return vectorized_data, vectorizer

def vectorize_n_kmeans(vectorized_data, vectorizer, n = 14):
    kmeans = KMeans(n_clusters=n).fit(vectorized_data)

    cluster_assignment = kmeans.predict(vectorized_data)

    centroids = kmeans.cluster_centers_

    thres = lambda x:  sorted(x[x > 0.05], reverse=True)

    cluster_words = vectorizer.inverse_transform(centroids)

    y = centroids * (centroids > 0.03)
    y2 = centroids * (centroids > 0.01) * (centroids < 0.03)

    top_words = vectorizer.inverse_transform(y)
    secondardy_words = vectorizer.inverse_transform(y2)

    return vectorized_data, kmeans, top_words, secondardy_words, cluster_assignment


def run_sentiment(novels):
    pos_score = []
    neg_score = []
    sid = SentimentIntensityAnalyzer()
    for novel_id in xrange(len(novels)):
        ss = sid.polarity_scores(novels[novel_id])
        pos_score.append(ss['pos'])
        neg_score.append(ss['neg'])

    with open('pos_score.dat', 'wb') as outfile:
        pickle.dump(pos_score, outfile, pickle.HIGHEST_PROTOCOL)

    with open('neg_score.dat', 'wb') as outfile:
        pickle.dump(neg_score, outfile, pickle.HIGHEST_PROTOCOL)

    return pos_score, neg_score

def run_k_means(vectorized_data, vectorizer, k_chosen_range = xrange(3,20)):
    top_words_dict = {}
    score = []
    cluster_assignment_dict = {}

    for k_chosen in k_chosen_range:
        tfidfed, kmeans, top_words, secondary_words, cluster_assignment = vectorize_n_kmeans(vectorized_data, vectorizer, n = k_chosen)
        top_words_dict[k_chosen] = top_words
        cluster_assignment_dict[k_chosen] = cluster_assignment
        score.append(kmeans.inertia_)
        for i in xrange(len(top_words)):
            text = ' '.join(top_words[i]) + ' ' + ' '.join(top_words[i]) + ' ' + ' '.join(secondary_words[i])
            run_word_cloud(text, i, k_chosen)
        print '\nCompleted Cluster {}:     Score: {}'.format(k_chosen, kmeans.inertia_)
    score_df = pd.DataFrame({'number_of_clusters': list(k_chosen_range), 'score': score})
    score_df.to_csv('score.csv')

    cluster_num = []
    cluster_id = []
    word_array = []
    for key in top_words_dict:
        for j in xrange(len(top_words_dict[key])):
            cluster_num.append(key)
            cluster_id.append(j)
            word_array.append(', '.join(top_words_dict[key][j]))

    word_df = pd.DataFrame({'number_of_clusters': cluster_num, 'cluster_id': cluster_id, 'top_words': word_array})
    word_df.to_csv('top_words.csv', encoding='utf-8')

    return cluster_assignment_dict

def reconst_mse(target, left, right):
    return (np.array(target - left.dot(right))**2).mean()

def describe_nmf_results(document_term_mat, W, H, component):
    sw = set(STOPWORDS)
    d = path.dirname(__file__)
    for topic_num, topic in enumerate(H):
        print 'Running Cluster: {} for N Components: {}'.format(topic_num, component)
        text1 = " ".join([feature_words[i] for i in topic.argsort()[:-5:-1]])
        text2 = " ".join([feature_words[i] for i in topic.argsort()[:-30:-1]])
        text = text1 + ' ' + text2
        '''
        wc = WordCloud(background_color="white", max_words=2000, stopwords=sw)
        # generate word cloud
        wc.generate(text)
        # store to file
        file_name = 'word_clouds/{}_cluster_{}.png'.format(component, topic_num)
        wc.to_file(path.join(d, file_name))
        '''
        #run_word_cloud(text, component, topic_num)
    return reconst_mse(document_term_mat, W, H)

def run_word_cloud(text, num_clusters = 5, cluster_num = 5, max_word = 8):
    d = path.dirname(__file__)
    sw = set(STOPWORDS)
    sw.update('shes', 'av', '8vo', 'youd', 'fourteen', 'yer', 'la', 'weve', \
            'nay', 'twas', 'sez', 'im', 'hed', 'ha', 'whats', 'er', 'wi', 'didn', \
            've', 'll', 'don', 'ain', 'couldn', 'doesn', 'wouldn', 'hadn', 'isn', \
            'hadnt', 'arent', 'youve', 'whats', 'shouldnt', 'theyre', 'maam', 'theyd', \
            'th', 'weve', 'fer', 'shes', 'hasnt','hadnt', 'un', 'hed', 'im', 'er', \
            'ah', 'ay', 'wi', 'youd', 'aint')

    querywords = text.split()

    resultwords  = [word for word in querywords if word.lower() not in sw]
    text = ' '.join(resultwords)

    wc = WordCloud(background_color="white", max_words= max_word, stopwords=sw)
    wc.generate(text)
    file_name = 'word_clouds/{}_cluster_{}.png'.format(cluster_num, num_clusters)
    wc.to_file(path.join(d, file_name))
    return

def get_terms_score(terms, vectorizer, X):
    idx = []
    for word in terms:
        try:
            idx.append(vectorizer.vocabulary_[word])
        except:
            continue
    return np.sum(X[:,idx],axis = 1)


def get_term_matrix(df, terms, vectorizer, X, pos, neg):
    net = pos - neg
    df_terms = pd.DataFrame({'title': df.title, 'author': df.author, 'year': df.year, 'net': net})
    for word in terms:
        try:
            df_terms[word] = X[:,vectorizer.vocabulary_[word]]
        except:
            continue
    df_terms.to_csv('term_matrix.csv')
    return df_terms


def random_text(df, start_words, num_clusters = 5, num_sentences = 100, len_sentences = 30):
    column_name = 'cluster_{}'.format(num_clusters)
    text = []
    cluster_num = []
    for c_num in xrange(num_clusters):
        print '\n \nGenerating sentences for cluster: {}\n'.format(c_num)
        df_c = df[df[column_name] == c_num].reset_index()
        rand_rows = np.random.randint(1, len(df_c), num_sentences)
        df_c = df_c.iloc[rand_rows,:]
        for file_name in df_c.file:
            file_ = open(file_name)
            markov = Markov.Markov(file_)
            try:
                out = markov.generate_markov_text(20, start_words)
                print '\n\t{}'.format(out)
                text.append(out)
                cluster_num.append(c_num)
            except:
                continue
    out_file_name = 'random_sentence_{}.csv'.format(start_words.strip())
    pd.DataFrame({'cluster_num': cluster_num, 'random_sentence': text}).to_csv(out_file_name)
    return

if __name__ == '__main__':
    #### read in the novel data
    df = pd.read_csv('allbookdata.csv')

    '''
    novels = [get_words(df.file[i]) for i in xrange(len(df))]
    with open('novels.dat', 'wb') as outfile:
        pickle.dump(novels, outfile, pickle.HIGHEST_PROTOCOL)
    '''
    '''
    with open('novels.dat', 'rb') as infile:
        novels = pickle.load(infile)
    '''
    '''
    novels_all_words = [get_words(df.file[i], True) for i in xrange(len(df))]
    with open('novels_all_words.dat', 'wb') as outfile:
        pickle.dump(novels_all_words, outfile, pickle.HIGHEST_PROTOCOL)
    '''
    ####

    #### tfidf the data
    '''
    vectorized_data, vectorizer = vectorize(novels)
    with open('vectorized_data.dat', 'wb') as outfile:
        pickle.dump(vectorized_data, outfile, pickle.HIGHEST_PROTOCOL)
    with open('vectorizer.dat', 'wb') as outfile:
        pickle.dump(vectorizer, outfile, pickle.HIGHEST_PROTOCOL)
    '''
    with open('vectorized_data.dat', 'rb') as infile:
        vectorized_data = pickle.load(infile)
    with open('vectorizer.dat', 'rb') as infile:
        vectorizer = pickle.load(infile)
    ####


    #### nmf
    error = []
    components = range(1,31)
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
    '''
    '''
    '''
    #### get sentiment analysis
    #pos_score, neg_score = run_sentiment(novels)
    with open('pos_score.dat', 'rb') as infile:
        pos_score = pickle.load(infile)
    with open('neg_score.dat', 'rb') as infile:
        neg_score = pickle.load(infile)
    ####

    #### k means
    k_chosen_range = xrange(4,8)
    cluster_assignment_dict = run_k_means(vectorized_data, vectorizer, k_chosen_range)
    ####

    #### create the output file
    output_df = pd.DataFrame({'book_title': df.title, 'author': df.author, 'year': df.year, \
            'year_range': df.year_range, 'nationality': df.nationality, 'pos_score': pos_score, \
            'neg_score': neg_score, 'file': df.file})

    for k_chosen in k_chosen_range:
        column_name = 'cluster_{}'.format(k_chosen)
        output_df[column_name] = cluster_assignment_dict[k_chosen]


    output_df.to_csv('clustering.csv', encoding='utf-8')
    #####

    output_df = pd.read_csv('clustering.csv')
    #### print random text
    #random_text(output_df, 'the man')
    #random_text(output_df, 'she said')
    #random_text(output_df, 'it is')
    #random_text(output_df, 'give me')
    random_text(output_df, 'likely that')
    random_text(output_df, 'is likely')
    random_text(output_df, 'how lucky')
    random_text(output_df, 'the probability')
    random_text(output_df, 'the chance')
    random_text(output_df, 'the bet')
    ####
    '''
    #df_terms = get_term_matrix(df, ['slaves', 'slavery', 'slave', 'negro', 'negroes', 'bet', 'probable', 'model', 'average', 'expectation', 'probability', 'likely', 'likelihood', 'statistic', 'statistics', 'normal', 'professional', 'assistant', 'lawyer', 'contract', 'trade', 'shaft', 'metal', 'rail', 'motor', 'coal', 'railroad', 'railway', 'machinery', 'chimney', 'manager', 'engine', 'electric', 'farmer', 'mill', 'potatoes', 'seed', 'corn', 'farm', 'plant', 'wheat', 'farmers', 'species', 'evolve', 'evolution'], vectorizer, vectorized_data.toarray(), pos_score, neg_score)

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


    #### wikipedia terms
    training_terms = ['farm', 'war', 'machine', 'family', 'philosophy', 'world war one', \
            'us civil war', 'art', 'sailor', 'feudalism', 'christianity', 'literature', \
            'science', 'factory', 'statistics', 'Race (human categorization)', 'probability', 'industry', 'novel', \
            'city', 'fiction', 'nation', 'nature', 'revolution', 'liberalism', 'evolution', \
            'sex', 'taste', 'Feelings', 'manual labor', 'democracy', 'poetry', 'Charity (practice)', 'capitalism', \
            'country']

    other_pg_texts = ['ebooks-unzipped/1228.txt', 'ebooks-unzipped/chance_and_luck.tex', 'ebooks-unzipped/euclid.tex', 'ebooks-unzipped/short_hist_math.tex']
    other_texts = [get_words(x) for x in other_pg_texts]

    wiki_documents = [get_wiki(t) for t in training_terms]

    documents = copy.copy(wiki_documents)
    for other in other_texts:
        documents.append(other)
    for novel in novels:
        documents.append(novel)
    ####

    #### cosine simliarity with wikipedia articles
    all_topics = vectorize(documents)[0]
    cosine_similarities = linear_kernel(all_topics, all_topics)
    ####



    all_terms = training_terms + other_pg_texts
    train_length = len(all_terms)
    training_category = []
    category_strength = []
    for i in xrange(len(df)):
        most_sim = np.argmax(cosine_similarities[i+train_length,0:train_length])
        max_sim = np.max(cosine_similarities[i+train_length,0:train_length])
        training_category.append(all_terms[most_sim])
        category_strength.append(max_sim)

    for i in xrange(len(all_terms)):
        output_df[all_terms[i]] = cosine_similarities[train_length:, i]

    #### term scores
    farm_score = get_terms_score(['farmer', 'mill', 'potatoes', 'seed', 'corn', 'farm', 'plant', 'wheat', 'farmers'], vectorizer, vectorized_data.toarray())
    factory_score = get_terms_score(['shaft', 'metal', 'rail', 'motor', 'coal', 'railroad', 'railway', 'machinery', 'chimney', 'manager', 'engine', 'electric'], vectorizer, vectorized_data.toarray())
    trade_score = get_terms_score(['professional', 'assistant', 'lawyer', 'contract', 'trade'], vectorizer, vectorized_data.toarray())
    statistics_score = get_terms_score(['bet', 'probable', 'model', 'average', 'expectation', 'probability', 'likely', 'likelihood', 'statistic', 'statistics', 'normal'], vectorizer, vectorized_data.toarray())
    darwin_score = get_terms_score(['species', 'evolve', 'evolution'], vectorizer, vectorized_data.toarray())
    slavery_score = get_terms_score(['slaves', 'slavery', 'slave', 'negro', 'negroes'], vectorizer, vectorized_data.toarray())
    ####
    '''
