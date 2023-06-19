from gensim.models import KeyedVectors
import os
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer
from math import log
from pymorphy2 import MorphAnalyzer
from string import punctuation
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from razdel import tokenize
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")

currdir = os.getcwd()
morph = MorphAnalyzer()
russian_stopwords = stopwords.words("russian")
path = os.path.join(currdir, 'data')

model_file = os.path.join(path, 'model.bin')
model = KeyedVectors.load_word2vec_format(model_file, binary=True)


def preprocess(text):
    tokens = tokenize(text.lower())
    tokens = [token.text for token in tokens 
              if token.text not in russian_stopwords
              and token.text not in punctuation + '«»...–—!?']
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    return lemmas


# Load database, preprocess, concatenate
def read_df(folder):
    queries = pd.read_excel(os.path.join(folder, 'queries_base.xlsx'),
                            usecols=[0, 1])
    queries.columns = ['Текст', 'Номер связки']
    answers = pd.read_excel(os.path.join(folder, 'answers_base.xlsx'))

    # Создаем базу вопросов и "чистый" корпус с леммами
    answers['Список вопросов'] = answers['Текст вопросов'].apply(
        lambda x: x.split('\n'))
    answers = answers.explode('Список вопросов')
    answers = answers.reset_index()
    queries = queries.dropna()
    answers = answers[['Список вопросов', 'Номер связки']]
    answers.rename(columns={'Список вопросов': 'Текст'}, inplace=True)

    # Объединяем базы
    df = pd.concat([answers, queries], ignore_index=True)
    test_size = round(queries.shape[0]*0.3)
    df['Preprocessed'] = df['Текст'].progress_apply(preprocess)
    df['n_words'] = df.Preprocessed.progress_apply(lambda x: len(x))
    df = df[df.n_words > 0]
    df['Prepstring'] = df.Preprocessed.progress_apply(lambda x: ' '.join(x))
    df = df.reset_index()
    return df, test_size


# Index corpus  and compute tf-idf
def tfidf_vectorizer(corpus, vectorizer):
    x = vectorizer.fit_transform(corpus)
    matrix = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
    matrix.to_pickle(os.path.join(path, "tfidf.pkl"))
    return matrix


# Bm25 score for each word in document
def bm25_score(id_word, id_doc, tfidf, avgdl, n, k, b):
    idf_score = idf(tfidf, id_word, n)
    tfidf_score = tfidf.item((id_doc, id_word))
    ld = np.count_nonzero(tfidf[id_doc, :])
    res = idf_score * ((tfidf_score * (k + 1))/(tfidf_score + k *
                                                (1 - b + b * (ld / avgdl))))
    return res


# idf for all documents
def idf(tfidf, id_word, n):
    n_qi = np.count_nonzero(tfidf[:, id_word])
    return log((n - n_qi + 0.5)/(n_qi + 0.5))


# Index corpus for bm25
def count_vec(corpus, vector):
    corpus = vector.fit_transform(corpus).toarray()
    with open(os.path.join(path, 'count_vectorizer_names.pkl'), 'wb') as fp:
        pickle.dump(vector.get_feature_names(), fp)
    return corpus


# Compute bm25 and download bm matrix
def bm25_matrix(tfidf, n, avgdl, k, b):
    n_words = tfidf.shape[1]
    bm = np.zeros((n, n_words))
    for id_doc in range(n):
        for id_word in range(n_words):
            bm[id_doc, id_word] = bm25_score(id_word, id_doc, tfidf,
                                             avgdl, n, k, b)
    return bm


# Normalize vector
def normalize_vec(v):
    return v / np.sqrt(np.sum(v ** 2))


# Create mean vectors for document
def doc2vec(lemmas):
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    vec = np.zeros((model.vector_size,))
    for idx, lemma in enumerate(lemmas):
        try:
            if lemma in model:
                lemmas_vectors[idx] = model[lemma]
        except AttributeError:
            continue
    if lemmas_vectors.shape[0] is not 0:
        vec = np.mean(lemmas_vectors, axis=0)
    if np.sum(vec) != 0:
        vec = normalize_vec(vec)
    return vec


# Create matrices for all documents
def create_doc_matrix(lemmas):
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    for idx, lemma in enumerate(lemmas):
        lemmas_vectors[idx] = model[lemma]
    if not lemmas:
        lemmas_vectors = np.zeros((1, model.vector_size))
    return normalize_vec(lemmas_vectors)


# Download constant values
def const_dict(data):
    n = data.shape[0]
    avgdl = sum(data.n_words) / n
    k = 2.0
    b = 0.75
    const = {'corpus_size': n, 'avgdl': avgdl, 'k': k, 'b': b}
    with open(os.path.join(path, 'const.json'), 'w') as fp:
        json.dump(const, fp)
    return n, avgdl, k, b


# Create matrix of vectorized documents
def get_doc2vec(data, n):
    matrix = np.zeros((n, model.vector_size))
    for idx, doc in enumerate(data.Preprocessed):
        matrix[idx] = doc2vec(doc).tolist()
    with open(os.path.join(path, 'doc2vec.pkl', 'wb')) as fp:
        pickle.dump(matrix, fp)
    return matrix


def get_doc2mat(data):
    all_matrices = []
    for doc in data.Preprocessed:
        all_matrices.append(create_doc_matrix(doc))
    with open(os.path.join(path, 'doc_matrices.pkl'), 'wb') as fp:
        pickle.dump(all_matrices, fp)
    return all_matrices


def main():
    df, test_size = read_df(os.path.join(path))
    train = df.iloc[:-test_size, :]
    train.to_excel('base.xlsx')
    n, avgdl, k, b = const_dict(train)
    count_vectorizer = CountVectorizer()
    cv = count_vec(train.Prepstring, count_vectorizer)
    vectorizer = TfidfVectorizer(stop_words=russian_stopwords)
    tfidf_vectorizer(train.Prepstring, vectorizer)
    bm25_matrix(cv, n, avgdl, k, b)
    get_doc2vec(train, n)
    get_doc2mat(train)


if __name__ == "__main__":
    main()
