import random, math, argparse

parser = argparse.ArgumentParser()
parser.add_argument('data', help = 'path to data')
parser.add_argument('output', help = 'path to output (csv)')
parser.add_argument('-l', '--lsa', type = int, default = 1, help = 'use lsa or not')

args = parser.parse_args()
title_path = args.data + '/title_StackOverflow.txt'
check_index = args.data + '/check_index.csv'
output = args.output
use_lsa = args.lsa
# title_path = 'data/title_StackOverflow.txt'
# check_index = 'data/check_index.csv'
# output = 'csv/kmeans_tfidf.csv'

from load_from_txt import load_data, load_check, calc_output
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

import numpy as np
from time import time

from nltk.corpus import stopwords

num_clusters = 85
n_components = 20
n_init = 100
max_df = 0.4
min_df = 2
# word_size = len(id2word)

# N = 0
# for doc in docs:
#     N += len(doc)

# # weight = []
# # for word in id2word:
# #     weight.append(word[])

# # titles = np.zeros((len(docs), word_size), dtype = np.int)
# vector = [0] * word_size
# titles = [vector] * len(docs)
# for index, doc in enumerate(docs):
#     print('index: ', index)
#     # line = np.zeros((word_size, ), dtype = np.int)
#     for id in doc:
#         # line[id] += 1
#         titles[index][id] += 1
#     # titles.append(line)

# # np_titles = np.array(titles)

word2id, id2word, docs, lines = load_data(title_path)

tfidf_vectorizer = TfidfVectorizer( max_df=max_df, max_features=None, # 200000,
                                    min_df=min_df, stop_words='english',
                                    use_idf=True)

# list_titles = titles.tolist()
tfidf_matrix = tfidf_vectorizer.fit_transform(lines)

if use_lsa:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    tfidf_matrix = lsa.fit_transform(tfidf_matrix)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()

km = KMeans(n_clusters=num_clusters, n_init=n_init, n_jobs = -1, verbose=1)

t1 = time()
km.fit(tfidf_matrix)
print("kmeans done in %fs" % (time() - t1))

clusters = km.labels_.tolist()

count = [0] * num_clusters
for c in clusters:
    count[c] += 1
print('count', count)

checks = load_check(check_index)
calc_output(clusters, checks, output)

