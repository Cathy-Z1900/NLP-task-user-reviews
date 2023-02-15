import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
import re


from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

PATH = "F:\data.csv"

file_object2 = open(PATH, encoding='utf-8', errors='ignore').read().split('\n')
data_set = []
for i in range(len(file_object2)):
    result = []
    seg_list = file_object2[i].split()
    for w in seg_list:
        result.append(w)
    data_set.append(result)


dictionary = corpora.Dictionary(data_set)
corpus = [dictionary.doc2bow(text) for text in data_set]
ldamodel = LdaModel(corpus, num_topics=10, id2word = dictionary, passes=30,random_state = 1)   #分为10个主题
print(ldamodel.print_topics(num_topics=10, num_words=15))

lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=30, random_state=1)
topic_list = lda.print_topics()

result_list=[]
for i in lda.get_document_topics(corpus)[:]:
    listj = []
    for j in i:
        listj.append(j[1])
    bz = listj.index(max(listj))
    result_list.append(i[bz][0])
print(result_list)
import pandas as pd

test=pd.DataFrame(data=topic_list)
print(test)
test.to_csv('topic2.csv')
import pyLDAvis.gensim_models
import pyLDAvis.gensim_models as gensimvis
import pickle
import pyLDAvis

data = gensimvis.prepare(lda, corpus, dictionary)
print(data)