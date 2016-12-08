import random, math, argparse

parser = argparse.ArgumentParser()
# parser.add_argument('data', help = 'path to data')
# parser.add_argument('output', help = 'path to output (csv)')
parser.add_argument('-l', '--lsa', type = int, default = 1, help = 'use lsa or not')
parser.add_argument('-c', '--num_clusters', type = int, default = 20, help = 'num_clusters')
parser.add_argument('-n', '--n_components', type = int, default = 20, help = 'n_components')

args = parser.parse_args()
# title_path = args.data + '/title_StackOverflow.txt'
# check_index = args.data + '/check_index.csv'
# doc_path = args.data + '/docs.txt'
# labels_path = args.data + '/label_StackOverflow.txt'
title_path = 'data/title_StackOverflow.txt'
check_index = 'data/check_index.csv'
doc_path = 'data/docs.txt'
labels_path = 'data/label_StackOverflow.txt'

# output = args.output
output = 'result.csv'
use_lsa = args.lsa
num_clusters = args.num_clusters
n_components = args.n_components
# title_path = 'data/title_StackOverflow.txt'
# check_index = 'data/check_index.csv'
# output = 'csv/kmeans_tfidf.csv'

from load_from_txt import load_data, load_labels, load_doc, load_check, calc_output
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import pairwise_distances_argmin
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
from time import time

# num_clusters = 70
# n_components = 20
n_init = 10
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

# word2id, id2word, docs, lines = load_data(title_path)
lines = load_doc(title_path)
docs = load_doc(doc_path)

tfidf_vectorizer = TfidfVectorizer( input=docs, max_df=max_df, max_features=None, # 200000,
                                    min_df=min_df, stop_words='english',
                                    use_idf=True)

# list_titles = titles.tolist()
tfidf_matrix = tfidf_vectorizer.fit_transform(lines)
print('shape: ', tfidf_matrix.shape)

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

km = KMeans(n_clusters=num_clusters, n_init=n_init, n_jobs = -1, verbose=0)

t1 = time()
km.fit(tfidf_matrix)
print("kmeans done in %fs" % (time() - t1))

print('shape: ', tfidf_matrix.shape)
print("Top terms per cluster:")
if use_lsa:
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
else:
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = tfidf_vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i, end = '')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end = '')
    print()

tsne = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
total = np.concatenate((tfidf_matrix, km.cluster_centers_), axis=0)
compressed = tsne.fit_transform(total)
matrix_2d = compressed[0:20000, ...]
center_2d = compressed[20000:, ...]

colors_ = colors.cnames.keys()

my_labels = km.labels_.tolist()
true_labels = load_labels(labels_path)


# temp_array = np.zeros((num_clusters, num_clusters), dtype=np.int)
# for i in range(20000):
#     temp_array[true_labels[i] - 1][my_labels[i]] += 1

# trans_array = [0] * num_clusters
# for i in range(num_clusters):
#     for j in range(1, num_clusters):
#         if temp_array[i][j] > temp_array[i][trans_array[i]]:
#             trans_array[i] = j

# print('trans_array: ', trans_array)

fig = plt.figure(figsize = (10, 10))
fig.subplots_adjust(left = 0.01, right = 0.99, bottom = 0.02, top = 0.9)
k_means_labels = np.array(true_labels)

ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(num_clusters), colors_):
    my_members = k_means_labels == k
    ax.plot(matrix_2d[my_members, 0], matrix_2d[my_members, 1], 'o', markerfacecolor = col,
            markersize = 2)
# for k, col in zip(range(num_clusters), colors_):
#     ax.plot(center_2d[trans_array[k]][0], center_2d[trans_array[k]][1], 'o', markerfacecolor = col,
#              markeredgecolor = 'k', markersize = 8)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.show()



# clusters = km.labels_.tolist()

# count = [0] * num_clusters
# for c in clusters:
#     count[c] += 1
# print('count', count)

# checks = load_check(check_index)
# calc_output(clusters, checks, output)

