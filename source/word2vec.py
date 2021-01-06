import sys
import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from annoy import AnnoyIndex
import pandas as pd

def load(filename):
    with open(filename, 'rb') as f:
        content = f.read()
    vocabulary_size = int.from_bytes(content[:4], byteorder='little')
    vocabulary = content[4:].split()[:vocabulary_size]
    print(vocabulary_size)
#    print(vocabulary)
    content = content[4 + vocabulary_size + sum(map(len, vocabulary)):]
    dim_embedding = int.from_bytes(content[:4], byteorder='little')
    print(dim_embedding)
    embeddings = np.zeros((vocabulary_size, dim_embedding))
    pos = 4
    unpack_arg = 'f' if sys.byteorder == 'little' else '>f'
    for i in range(vocabulary_size):
        for j in range(dim_embedding):
            embeddings[i][j] = struct.unpack(unpack_arg, content[pos:pos+4])[0]
            pos += 4
    return vocabulary, embeddings

def show_tsne_graph(vocabulary, embeddings):
    result = TSNE(n_components=2).fit_transform(embeddings)
    indices = list(range(embeddings.shape[0]))
    plt.scatter(result[indices, 0], result[indices, 1])
    for i, word in enumerate(vocabulary):
        plt.annotate(word, xy=(result[indices[i], 0], result[indices[i], 1]))
    plt.show()

def nearest_neighbors(vocabulary, embeddings, k=40, metric='angular'):
    annoy_index = AnnoyIndex(embeddings.shape[1], metric=metric)
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)
    annoy_index.build(embeddings.shape[1])
    word_to_ix = {word.decode(): i for i, word in enumerate(vocabulary)}
    ix_to_word = {i: word.decode() for i, word in enumerate(vocabulary)}
    while True:
        print('Enter word (EXIT to break): ')
        word = input()
        if word == 'EXIT': break
        if word not in word_to_ix:
            print('Out of dictionary word!')
        else:
            nn_res = annoy_index.get_nns_by_vector(embeddings[word_to_ix[word]], k + 1, include_distances=True)
            df = pd.DataFrame({'word': map(lambda x: ix_to_word[x], nn_res[0][1:]),
                               'distance': map(lambda x: x, nn_res[1][1:])})
            print(df)

def analogy(vocabulary, embeddings, k=4, metric='angular'):
    # Example input: paris france berlin
    annoy_index = AnnoyIndex(embeddings.shape[1], metric=metric)
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)
    annoy_index.build(embeddings.shape[1])
    word_to_ix = {word.decode(): i for i, word in enumerate(vocabulary)}
    ix_to_word = {i: word.decode() for i, word in enumerate(vocabulary)}
    while True:
        print('Enter three words (EXIT to break): ')
        words = input().split()
        if words and words[0] == 'EXIT': break
        if not all(word in word_to_ix for word in words):
            print('Out of dictionary word!')
            continue
        print(words)
        vec1 = embeddings[word_to_ix[words[0]]]
        vec2 = embeddings[word_to_ix[words[1]]]
        vec3 = embeddings[word_to_ix[words[2]]]
        nn_res = annoy_index.get_nns_by_vector(vec3 + (vec2 - vec1), k, include_distances=True)
        df = pd.DataFrame({'word': map(lambda x: ix_to_word[x], nn_res[0]),
                           'distance': map(lambda x: x, nn_res[1])})
        print(df)
