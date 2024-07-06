import json
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textstat

import nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
# from wordcloud import WordCloud, STOPWORDS
def extract_features(text):
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    word_count = len(words)
    sentence_count = len(sentences)
    lexical_diversity = len(set(words)) / word_count if word_count != 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count != 0 else 0
    not_stopword_ratio = len([word for word in words if word.lower() not in stop_words]) / word_count if word_count != 0 else 0
    punctuation_count = len([char for char in text if char in string.punctuation])
    avg_word_length = np.mean([len(word) for word in words]) if word_count != 0 else 0
    
    features = {
        'WordCount': word_count,
        'LexicalDiversity': lexical_diversity,
        'NotStopwordRatio': not_stopword_ratio,
        'PunctuationCount': punctuation_count,
        'AvgWordLength': avg_word_length,
        'AvgSentenceLength': avg_sentence_length,
        'ReadingEase': textstat.flesch_reading_ease(text),
    }
    return features


def load_dataset(path, test=True):
    '''Convert samples in JSON to dataframe
    0 if the text is AI-generated
    1 if the text is human-generated
    '''
    data = []
    columns = ['id', 'text', 'label']
    with open(path) as f:
        lines = f.readlines()        
        if test:
            for line in lines:
                line_dict = json.loads(line)
                features = extract_features(line_dict['text'])
                data.append([line_dict['id'], line_dict['text'], line_dict['label']] + list(features.values()))
            feature_columns = list(features.keys())
            columns.extend(feature_columns)
        else:
            columns = columns[:-1] + list(features.keys())
            for line in lines:
                line_dict = json.loads(line)
                features = extract_features(line_dict['text'])
                data.append([line_dict['id'], line_dict['text']] + list(features.values()))

    return pd.DataFrame(data, columns=columns).set_index('id')


def get_text_len(row):
    return len(row['text'].split(' '))

# def plot_word_cloud(df):
#     '''Plot word cloud of dataframe content'''
#     # Without stop words
#     word_cloud = WordCloud(width=800, height=800, background_color='white', stopwords=STOPWORDS).generate(" ".join(df['text']))

#     plt.figure(figsize=(6,6))
#     plt.imshow(word_cloud)
#     plt.axis('off') 
#     plt.tight_layout()
#     plt.show()

def get_top_ngram(text, n=3, top=10):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=n).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:top]

    top_n_bigrams=_get_top_ngram(text,n)[:top]
    x,y=map(list,zip(*top_n_bigrams))
    return x