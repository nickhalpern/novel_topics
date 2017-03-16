import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import time


def scrape_year_wiki():
    year_range = range(1800,1981)
    title = []
    url = []
    year_list = []
    for year in year_range:
        novel_txt = 'https://en.wikipedia.org/wiki/Category:{}_novels'.format(year)
        r = requests.get(novel_txt)
        soup = BeautifulSoup(r.text, 'html.parser')
        try:
            mw_pages = soup.find_all('div', {"id": "mw-pages"})[0]
            li = mw_pages.find_all('li')
            for i in li:
                year_list.append(year)
                title.append(i.a.get('title'))
                url.append(i.a.get('href'))
            print 'Novels loaded for {}'.format(year)
        except:
            print 'No novels in {}'.format(year)
    A = pd.DataFrame({'year': year_list, 'title': title, 'url': url})
    A.to_csv('wikipedia_book_data.csv', encoding='utf-8')
    return A


def scrape_book_data(url):
    novel_url = 'https://en.wikipedia.org{}'.format(url)
    r = requests.get(novel_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    mw_pages = soup.find('table')
    table = mw_pages.find_all('th')
    table_contents = mw_pages.find_all('td')

    book_attrib = []
    book_values = []
    for i in xrange(len(table)):
        if len(table[i].contents) == 3:
            attrib = ''.join(table[i].contents[1].contents)
        else:
            attrib = ''.join(table[i].contents)
        try:
            value = table_contents[i+1].a.get('title')
        except:
            value = ''.join(table_contents[i+1])
        #print '{}: {}'.format(attrib, value)
        book_attrib.append(attrib)
        book_values.append(value)

    return book_attrib, book_values


def get_book_details(df, file_name = 'wikipedia_book_detailed_data'):
    book_url = []
    book_attrib = []
    book_values = []

    for url in df['url']:
        print 'scraping url: {}'.format(url)
        try:
            atb, val = scrape_book_data(url)
            for a in atb:
                book_url.append(url)
                book_attrib.append(a)
            for v in val:
                book_values.append(v)
        except:
            pass

    A = pd.DataFrame({'url': book_url, 'attribute': book_attrib, 'value': book_values})
    file_string = '{}.csv'.format(file_name)
    A.to_csv(file_string, encoding='utf-8')
    return A

def get_goog_publish_data(df):
    df = df.reset_index()
    df = df[['title', 'author']]
    date = []
    title_list = []
    count = 1
    df['term1'] = ['{}+{}'.format(df.iloc[i,0].replace(' ', '+').replace(',', ''), df.iloc[i,1].split(' ')[-1]) for i in xrange(len(df))]
    df['term2'] = [df.iloc[i,0].replace(' ', '+').replace(',', '') for i in xrange(len(df))]
    for i in xrange(len(df)):
        term1 = df.term1[i]
        term2 = df.term2[i]
        title = df.title[i]
        novels = 'https://www.google.com/search?q={}+originally+published'.format(term1)
        r = requests.get(novels)
        soup = BeautifulSoup(r.text, 'html.parser')
        try:
            vbb = soup.find_all('div', {"id": "_vBb"})[0]

            try:
                contents = vbb.find_all('span')[0].contents
            except:
                contents = vbb.find_all('span')[0]

            print '{}/{}: Book: {}   Year: {}'.format(count, len(df.title), title, contents)

            date.append(contents)
            title_list.append(title)
        except:
            date.append([0])
            title_list.append(title)
            print '{}/{}: Could not scrape: {}'.format(count, len(df.title), title)

        time.sleep(1)#np.random.randint(1,15))
        if count/float(5) == int(count/5):
            A1 = pd.read_csv('google_scrape.csv')
            A1 = A1.drop_duplicates(subset = ['date', 'title'])
            A1 = A1[['title', 'date']]
            A2 = pd.DataFrame({'title': title_list, 'date': date})
            A = pd.concat([A1, A2])
            #A = A.drop_duplicates(subset = ['date', 'title'])
            A.to_csv('google_scrape.csv', encoding='utf-8')
            #with open('google_scrape.csv', 'a') as f:
            #    A.to_csv(f, header=False)
            time.sleep(120)
            title_list = []
            date = []
        count += 1
    return date, title_list


def clean_wiki(wiki_array):
    yr = []
    for dt in wiki_array:
        found = 0
        if type(dt) is list:
            dt = str(dt)
            found = 1
        if type(dt) is str:
            dt_str = dt.replace('-', ' ').split(' ')
            for chars in dt_str:
                if (len(chars) == 4) & ((chars[0] == '1') | (chars[0] == '2')) & (found == 0):
                    found = 1
                    yr.append(int(chars))
        if found == 0:
            yr.append(0)
    return yr

def clean_gut(gut_array):
    yr = []
    for dt in gut_array:
        try:
            yr.append(int(dt))
        except:
            yr.append(None)
    return yr


def clean_goog(goog_array):
    yr = []

    for date in goog_array:
        success = 0
        try:
            dt_string = str(date).split('[')[-1].split("u'")[-1].split(']')[0].split("'")[0].split(' ')[-1]
            '''
            for dt in dt_string.split(' '):
                if (len(dt) == 4) & ((dt[0] == '1') | (dt[0] == '2')):
                    yr.append(dt)
            '''
            yr.append(dt_string)
            success = 1

        except:
            pass

        if success == 0:
            yr.append(None)

    return yr


'''
def clean_goog2(goog_array):
    yr = []
    for dt in goog_array:
        try:
            dt_string = str(x).split('[')[1].split(']')[0].split('u')[-1]
            for dt in dt_string.split(' '):
                if (len(dt) == 4) & ((dt[0] == '1') | (dt[0] == '2')):
                    yr.append(dt)
        except:
            yr.append(None)
    return yr
'''

def get_year(r):
    if r.dt_w1 > 1000:
        return r.dt_w1
    elif r.dt_w2 > 1000:
        return r.dt_w2
    elif r.dt_g > 1000:
        return r.dt_g
    #elif r.dt_gut > 1000:
    #    return r.dt_gut
    else:
        return None



if __name__ == '__main__':
    df_author = pd.read_csv('list_of_novelists.csv', sep = '\t')
    df_books = pd.read_csv('book_data.csv')
    df_wiki = pd.read_csv('wikipedia_book_data.csv')
    df_wiki2 = pd.read_csv('wikipedia_book_detailed_data.csv')
    df_wiki3 = pd.read_csv('wikipedia_imputed_url_data.csv')
    df_author.columns = ['author', 'nationality']
    df = pd.merge(df_author, df_books, on = 'author')
    df['url'] = ['/wiki/{}'.format(str(z).replace(' ', '_')) for z in df.title]
    df1 = df.merge(df_wiki, how = 'left', on = 'url')
    df2 = df1.merge(df_wiki2[df_wiki2.attribute == 'Publication date'], how = 'left', on = 'url')
    df3 = df2.merge(df_wiki3[df_wiki3.attribute == 'Publication date'], how = 'left', on = 'url')


    rnd = 0
    max_i = 50
    while rnd <= max_i:
        df_goog = pd.read_csv('google_scrape.csv')
        df4 = df3.merge(df_goog, how = 'left', left_on = 'title_x', right_on ='title')
        df4 = df4[['author', 'file', 'title_x', 'year_x', 'value_x', 'value_y', 'date']]
        df4.columns = ['author', 'file', 'title', 'dt_gut', 'dt_w1', 'dt_w2', 'dt_g']
        df4['dt_gut'] = clean_gut(df4['dt_gut'])
        df4['dt_w1'] = clean_wiki(df4['dt_w1'])
        df4['dt_w2'] = clean_wiki(df4['dt_w2'])
        df4['dt_g'] = clean_goog(df4['dt_g'])

        y = []
        for i in xrange(len(df4)):
            r = df4.iloc[i,:]
            y.append(get_year(r))

        df_for_google = df4[df4.dt_g.map(lambda x: x == 'nan')]
        rand_rows = np.random.randint(1, len(df_for_google), 25)
        df_for_google = df_for_google.iloc[rand_rows,:]
        date, title_list = get_goog_publish_data(df_for_google[['title', 'author']])
        rnd += 1
        time.sleep(10*60)
        print '{}/{} rounds completed'.format(rnd, max_i)


    df_goog = pd.read_csv('google_scrape.csv')
    df4 = df3.merge(df_goog, how = 'left', left_on = 'title_x', right_on ='title')
    df4 = df4[['author', 'file', 'title_x', 'year_x', 'value_x', 'value_y', 'date']]
    df4.columns = ['author', 'file', 'title', 'dt_gut', 'dt_w1', 'dt_w2', 'dt_g']
    df4['dt_gut'] = clean_gut(df4['dt_gut'])
    df4['dt_w1'] = clean_wiki(df4['dt_w1'])
    df4['dt_w2'] = clean_wiki(df4['dt_w2'])
    df4['dt_g'] = clean_goog(df4['dt_g'])

    y = []
    for i in xrange(len(df4)):
        r = df4.iloc[i,:]
        y.append(get_year(r))
    df4['year'] = y

    pd.to_numeric(df4['year'], errors='coerce')

    df5 = df4[df4.year != 'nan']
    df5['year'] = df5['year'].astype('int')
    df5 = df5.drop_duplicates(subset = ['author', 'title'])

    novelists = pd.read_csv('list_of_novelists.csv', sep = '\t')
    novelists.columns = ['author', 'nationality']
    all_data = pd.merge(df5, novelists, on = 'author')

    bins = [1000, 1800, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 3000]
    year_range = pd.cut(all_data['year'], bins)
    year_range = [year_range[i].split('(')[1].split(',')[1].strip()[0:4] for i in xrange(len(year_range))]
    all_data['year_range'] = year_range

    all_data[['author', 'title', 'file', 'nationality', 'year', 'year_range']].to_csv('allbookdata.csv')


    #plt.hist(df4['year'], bins = range(1800,1980,10))
    #plt.show()



    '''
    df_1['url'] = ['/wiki/{}'.format(z.replace(' ', '_')) for z in df_1.title]
    get_book_details(df_1, 'wikipedia_imputed_url_data')
    df_imputed = pd.read_csv('wikipedia_imputed_url_data.csv')
    df_imputed[df_imputed.attribute == 'Publication date']
    '''

    '''
    df_wiki = scrape_year_wiki()
    df_wiki['title'] = [title.split('(')[0] for title in df_wiki['title']]
    df_books['title'] = [title.split('(')[0] if type(title) == 'str' else title for title in df_books['title']]
    df = pd.merge(df_wiki, df_books, on = 'title')

    #plt.hist(df['year_x'], bins = range(1800,1980,10))
    #plt.show()
    get_book_details(df)
    '''

    '''
    look for volume #, vol., v1..       , book #, delete where we have complete
    '''

    #isbn key: 3HQHJ0XY
