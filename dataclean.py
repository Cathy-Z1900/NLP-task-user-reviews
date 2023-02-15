import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import re
import string
df=pd.read_csv("reviews.csv")
df_drop=df.dropna(axis=0,how="any")
rev=df_drop
a=df_drop["reviewerID"].value_counts()
pd.set_option('display.max_columns', None)
text=rev['reviewText']

def clean_text_round1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)
data_clean_1 = pd.DataFrame(text.apply(round1))

def clean_text_round2(text):
  text = re.sub('[‘’“”…]', '', text)
  text = re.sub('\n', '', text)
  return text

round2 = lambda x: clean_text_round2(x)
data_clean_2 = pd.DataFrame(data_clean_1['reviewText'].apply(round2))

import string
from string import punctuation

def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct
data_clean_2['reviewText_pun']=data_clean_2['reviewText'].apply(lambda x: remove_punctuation(x))

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
sw_nltk = set(stopwords.words('english'))

def remove_stopwords(text):
    text=[word for word in text.split() if word not in sw_nltk]
    return text
data_clean_2['reviewText_pun_stop']= data_clean_2['reviewText_pun'].apply(lambda x: remove_stopwords(x))
pd.set_option('display.max_columns', None)
bb=data_clean_2.dropna(how="any")

