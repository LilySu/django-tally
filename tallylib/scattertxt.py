import re
import sys
from datetime import datetime
import pandas as pd
import spacy
import scattertext as st
import numpy as np
import math
# Local imports
# from tallylib.scraper import yelpScraper # Deleted on 2020-01-13
from tallylib.sql import getLatestReviews

# viztype0 (Top 10 Positive/Negative Phrases)
def getReviewPosNegPhrases(df_reviews, topk=10):

    if df_reviews.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_reviews.copy()
    df['stars'] = df['stars'].astype(str)

    nlp = spacy.load("en_core_web_sm")
    nlp.Defaults.stop_words |= {'will','because','not','friends',
    'amazing','awesome','first','he','check-in', 'and', 'some',
    '=','= =','male','u','want', 'u want', 'cuz', 'also', 'find',
    'him',"i've", 'deaf','on', 'her','told','told him',
    'ins', 'check-ins','check-in','check','I', 'i"m', 
    'i', ' ', 'it', "it's", 'it.','they','coffee','place', "it 's", "'s", 
    'they', 'the', 'this','its', 'l','-','they','this',
    'don"t','the ', ' the', 'it', 'i"ve', 'i"m', '!', '&',
    '1','2','3','4', '5','6','7','8','9','0','/','.',','}

    corpus = st.CorpusFromPandas(df,
                                 category_col='stars',
                                 text_col='text',
                                 nlp=nlp).build()
    term_freq_df = corpus.get_term_freq_df()

    categories = df['stars'].unique()
    high, poor = np.array([]), np.array([])
    if '5' in categories:
        high = corpus.get_scaled_f_scores('5')
    elif '4' in categories:
        high = corpus.get_scaled_f_scores('4')
    if '1' in categories:
        poor =  corpus.get_scaled_f_scores('1')
    elif '2' in categories:
        poor = corpus.get_scaled_f_scores('2')

    df_high, df_poor = pd.DataFrame(), pd.DataFrame()
    columns = ['term', 'score']
    if high.shape[0] > 0:
        df_high = pd.DataFrame([term_freq_df.index.tolist(), high]).T
        df_high = df_high.sort_values(1, ascending=False).head(topk)
        df_high.columns = columns
    if poor.shape[0] > 0:
        df_poor = pd.DataFrame([term_freq_df.index.tolist(), poor]).T
        df_poor = df_poor.sort_values(1, ascending=False).head(topk)
        df_poor.columns = columns

    # positive dataframe, negative dataframe 
    return df_high.head(topk), df_poor.tail(topk)

# viztype0 updated 1-22-20 (Top 10 Positive/Negative Long 6-word Phrases)
def getPosNegLongPhrases(df_reviews, topk=10):
    nlp = spacy.load("en_core_web_sm")
    if df_reviews.empty:
        return pd.DataFrame()
    df = df_reviews.copy()
    df['stars'] = df['stars'].astype(str)
    df = df.dropna()
    df['only_alphabets'] = df['text'].apply(lambda x: ' '.join(re.findall("[a-zA-Z]+", x)))

    replace_dict_phrase_count = {'[':'',']':'','-':'','!':'','.':'',"'":''}
    for key in replace_dict_phrase_count.keys():
        df['only_alphabets'] = df['only_alphabets'].str.replace(key, replace_dict_phrase_count[key])
        df['only_alphabets'] = df['only_alphabets'].str.lower()

    stopwords = ['"','+','@','&','*','\\',')','(','\(','\xa0','0','1','2','3','4','5','6','7','8',
    '9','/','$',"'d","'ll","'m",'+','maybe','from','first','here','only','put','where','got','sure',
    'definitely','food','yet','our','go','since','really','very','two',"n't",'with','if',"'s",'which',
    'came','all','me','(',')','makes','make','were','immediately','get','been','ahead','also','that',
    'one','have','see','what','to','we','had','.',"'re",'it','or','he','she','we','us','how','went',
    'no','"','of','has','by','bit','thing','place','so','ok','and','they','none','was','you',"'ve",
    'did','be','and','but','is','as','&','you','has','-',':','and','had','was','him','so','my','did',
    'would','her','him','it','is','by','bit','thing','place','[',']','while','check-in','=','= =',
    'want', 'good','husband', 'want','love','something','your','they','your','cuz','him',"i've",'her',
    'told', 'check', 'i"m', "it's",'they', 'this','its','they','this',"don't",'the',',', 'it', 'i"ve',
     'i"m', '!', '1','2','3','4', '5','6','7','8','9','0','/','.']
    def filter_stopwords(text):
        for i in str(text):
            if i not in stopwords:
                return str(text)

    # if item in stopwords list partially matches, delete, single letters like 'i' would be deleted 
    # from inside individual words if in list
    df = df[~df['only_alphabets'].isin(stopwords)]
    # if the following words fully matches, filter out
    full_match_list = ['i','a','an','am','at','are','in','on','for','','\xa0\xa0','\xa0','\(']
    df = df[~df['only_alphabets'].isin(full_match_list)]
    try: 
        corpus = st.CorpusFromPandas(df,
                                    category_col='stars',
                                    text_col='only_alphabets',
                                    nlp=nlp).build()
        term_freq_df = corpus.get_term_freq_df()
        term_freq_df = pd.DataFrame(term_freq_df.to_records())# flatten multi-level index to rename columns
        term_freq_df = term_freq_df.rename(columns = {'5 freq': '5.0', '4 freq': '4.0','2 freq': '2.0', 
        '1 freq': '1.0' })

        categories = df['stars'].unique()
        freq_word_list = np.array([])
        if '5' in categories:
            freq_word_list  = corpus.get_scaled_f_scores('5')
        elif '4' in categories:
            freq_word_list  = corpus.get_scaled_f_scores('4')
        if '1'  in categories:
            freq_word_list  =  corpus.get_scaled_f_scores('1')
        elif '2' in categories:
            freq_word_list  = corpus.get_scaled_f_scores('2')

        df_wordFreq = pd.DataFrame()
        columns = ['term', 'score']
        if freq_word_list.shape[0] > 0:
            df_wordFreq = pd.DataFrame([term_freq_df.term.tolist(), freq_word_list ]).T
            df_wordFreq = df_wordFreq.sort_values(1, ascending=True)#.head(topk)
            df_wordFreq.columns = columns
    except: 
        df['word_list'] = df['only_alphabets'].apply(lambda x: x[1:-1].split(' '))
        df['word_list'] = df['word_list'].astype(str)

        df['word_list'] = df['word_list'].apply(lambda x: ''.join([str(i) for i in x]))
        df['word_list'] = df['word_list'].str.lower()

        df_wordFreq = df[['word_list', 'stars']]

        s= df_wordFreq.apply(lambda x: pd.Series(x['word_list']),axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'word_list'

        df_wordFreq = pd.DataFrame(df['word_list'].str.split(',').tolist(), index=df['stars']).stack()

        df_wordFreq = df_wordFreq.reset_index()[[0, 'stars']] # var1 variable is currently labeled 0
        df_wordFreq.columns = ['term', 'score'] # renaming var1
        df_wordFreq = df_wordFreq.reset_index(drop=False)

        replace_dict_phrase_count = {'[':'',']':'','-':'','!':'','.':'',"'":'', ' ':''}
        for key in replace_dict_phrase_count.keys():
            df_wordFreq['term'] = df_wordFreq['term'].str.replace(key, replace_dict_phrase_count[key])
            df_wordFreq['term'] = df_wordFreq['term'].str.lower()
    x, y = df_wordFreq.shape
    if x > 100:
        df_wordFreq = pd.concat([df_wordFreq.head(50), df_wordFreq.tail(50)])
    x, y = df_wordFreq.shape# updated size
    top_terms_list = []
    for i in range(math.ceil(x/2)):
        try:
            new_df = df[df['only_alphabets'].str.contains(df_wordFreq['term'].iloc[i])]#if word appears 
            # in review, create a dataframe with each row being the word occurring in a different review
            pos_first_df = new_df.sort_values(by='stars', ascending=False)#rank the dataframe with most 
            # positive reviews first
            if pos_first_df['text'].iloc[0] not in top_terms_list:#get the highest star rating review
                top_terms_list.append(pos_first_df['text'].iloc[0])
        except IndexError as e:
            pass
    worst_terms_list = [] 
    for i in reversed(range(math.ceil(x/2), x)):
        try:
            new_df = df[df['only_alphabets'].str.contains(df_wordFreq['term'].iloc[i])]#if word appears 
            # in review, create a dataframe with each row being the word occurring in a different review
            neg_first_df = new_df.sort_values(by='stars', ascending=True)#rank the dataframe with worst 
            # reviews first
            if neg_first_df['text'].iloc[0] not in worst_terms_list:#get the lowest star rating review
                worst_terms_list.append(neg_first_df['text'].iloc[0])#prevent duplicates
        except IndexError as e:
            pass
    del [df]
    negative_list = []
    for i in reversed(range(math.ceil(x/2), x)):
        for list_of_words in worst_terms_list:
            word_list = list_of_words.split(' ')
            for word in word_list:
                try: 
                    if df_wordFreq['term'].iloc[i] == word: # find word occurrence in original 
                        # comma separated word list of reviews
                        try:
                            index = word_list.index(word)
                            string_from_phrases = ' '.join(word_list[max(0,index-2):min(index+4, 
                            len(word_list))])
                            negative_list.append(string_from_phrases)
                        except ValueError as e:
                            pass
                except IndexError as e:#if there are less than the last half of 
                        # the df_wordFreq words fter stopword filtering, just get the first word and 
                        # its occurrence in the original review
                    if df_wordFreq['term'].iloc[0] == word:
                        try:
                            index = word_list.index(word)
                            string_from_phrases = ' '.join(word_list[max(0,index-2):min(index+4, 
                            len(word_list))])
                            negative_list.append(string_from_phrases)
                        except ValueError as e:
                            pass
    negative_df = pd.DataFrame(negative_list)
    negative_df = negative_df.reset_index(drop=False)
    negative_df = negative_df.rename(columns={'index':'score', 0 : 'term'})
    neg_no_dup = negative_df.drop_duplicates(subset='term')
    negative_phrase_list, y = neg_no_dup.shape
    if negative_phrase_list <= 10:
        num_time_append = 10 - negative_phrase_list
        for i in range(num_time_append):
            if 'term' not in list(negative_df):
                negative_df = negative_df.append(pd.DataFrame([.5], columns=['score']))
                negative_df['term'] = ''
            negative_df = negative_df.append(pd.DataFrame([[.5, '']], columns=['score', 'term']))
    else:
        negative_df = neg_no_dup
        del [neg_no_dup]
    replace_dict_phrase = {',':' ','\u00a0':'','\n':'','!':'','.':'',"'":''}
    for key in replace_dict_phrase.keys():
        negative_df['term'] = negative_df['term'].str.replace(key, replace_dict_phrase[key])
    #normalize score for positive connotation words going from 0 to 0.5
    negative_df['score'] = negative_df['score'].div((negative_df['score'].max())*2, axis=0)
    negative_df = negative_df.sort_values(by=['score'],ascending = False)
    negative_df['score'] = negative_df['score'].round(decimals=4)


    positive_list = []
    for i in range(math.ceil(x/2)):
        for list_of_words in top_terms_list:
            word_list = list_of_words.split(' ')
            for word in word_list:
                try: 
                    if df_wordFreq['term'].iloc[i] == word:# find word occurrence in original 
                        # comma separated word list of reviews
                        try:
                            index = word_list.index(word)
                            string_from_phrases = ','.join(word_list[max(0,index-2):min(index+4, 
                            len(word_list))])
                            positive_list.append(string_from_phrases)
                        except ValueError as e:
                            pass
                except IndexError as e:
                    if df_wordFreq['term'].iloc[0] == word:#if there are less than the first half of 
                        # the df_wordFreq words fter stopword filtering, just get the first word and 
                        # its occurrence in the original review
                        try:
                            index = word_list.index(word)
                            string_from_phrases = ','.join(word_list[max(0,index-2):min(index+4, 
                            len(word_list))])
                            positive_list.append(string_from_phrases)
                        except ValueError as e:
                            pass
    positive_df = pd.DataFrame(positive_list)
    positive_df = positive_df.reset_index(drop=False)
    positive_df = positive_df.rename(columns={'index':'score', 0 : 'term'})
    pos_no_dup = positive_df.drop_duplicates(subset='term')
    positive_phrase_list, y = pos_no_dup.shape
    if positive_phrase_list <= 10:
        num_time_append = 10 - positive_phrase_list
        for i in range(num_time_append):
            if 'term' not in list(positive_df):
                positive_df = positive_df.append(pd.DataFrame([.5], columns=['score']))
                positive_df['term'] = ''
            positive_df = positive_df.append(pd.DataFrame([[.5, '']], columns=['score', 'term']))
    else:
        positive_df = pos_no_dup
        del [pos_no_dup]
    for key in replace_dict_phrase.keys():
        positive_df['term'] = positive_df['term'].str.replace(key, replace_dict_phrase[key])
    #normalize score for positive connotation words going from 0.5 to 1 
    positive_df['score'] = positive_df['score'].div(((positive_df['score'].max())*2), axis=0)+0.5
    positive_df = positive_df.sort_values(by=['score'],ascending = False)
    positive_df['score'] = positive_df['score'].round(decimals=4)
    
    return positive_df.head(topk), negative_df.tail(topk)


# viztype3
def getYelpWordsReviewFreq(df_reviews):
  
    if df_reviews.empty:
        return pd.DataFrame()

    df = df_reviews.copy()

    df.columns = ['date', 'text', 'stars']
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['week_number_of_year'] = df['date'].dt.week
    df = df.groupby(['year', 'month','week_number_of_year']).mean()
    df = pd.DataFrame(df.to_records()) # flatten groupby column
    df = df.iloc[::-1].head(8)
    df['cumulative_avg_rating'] = df['stars'].mean()

    # get the date of last day of the week
    list = []
    for _, row in df.iterrows():
        text = str(row['year'].astype(int)) + '-W' + \
               str(row['week_number_of_year'].astype(int)) + '-6'
        date_of_week = datetime.strptime(text, "%Y-W%W-%w").strftime('%Y-%m-%d')
        list.append(date_of_week)
    df['date_of_week'] = list
    df = df.iloc[::-1]

    return df


def getDataViztype0(business_id):
    ''' Deleted on 2020-01-13
    # do web scraping 
    yelpScraperResult = yelpScraper(business_id)
    '''
    data = getLatestReviews(business_id, limit=200)
    if len(data)==0:
        return {}
    df_reviews = pd.DataFrame(data, columns=['date', 'text', 'stars'])
    del data
    df_reviews['date'] = pd.to_datetime(df_reviews['date'])

    # viztype0
    df_positive, df_negative = getPosNegLongPhrases(df_reviews)
    positive, negative = [], []
    if not df_positive.empty:
        positive = [{'term': row[0], 'score': row[1]} 
            for row in df_positive[['term', 'score']].values]
    if not df_negative.empty:
        negative = [{'term': row[0], 'score': row[1]} 
            for row in df_negative[['term', 'score']].values]
    viztype0 = {
        'positive': positive, 
        'negative': negative
    }
    del [df_positive, df_negative]

    # viztype3
    df_bydate = getYelpWordsReviewFreq(df_reviews)
    viztype3 = {}
    if not df_bydate.empty:
        viztype3 = {
            'star_data': [{'date': row[0], 
                           'cumulative_avg_rating': row[1], 
                           'weekly_avg_rating': row[2]}
            for row in df_bydate[['date_of_week', 'cumulative_avg_rating', 'stars']].values]
        }
    del [df_bydate]

    # API data formatting
    results = {
        'viztype0': viztype0,
        'viztype3': viztype3
        }

    return results


