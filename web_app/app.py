from flask import Flask
from flask import render_template, request
import os
import pickle
import numpy as np
from pymorphy2 import MorphAnalyzer
from string import punctuation
import pandas as pd
from razdel import tokenize
from nltk.corpus import stopwords
import nltk
import re
from bs4 import BeautifulSoup
import pymongo
import pandas as pd
from razdel import tokenize
from razdel import sentenize
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import os
from bson import ObjectId
from navec import Navec
from slovnet import NER


nltk.download("stopwords")
morph = MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

# Загрузка базы
client = pymongo.MongoClient('localhost', 27017, unicode_decode_error_handler='ignore')
db = client['mongo_for_folklore']
text_collection = db.texts # инфа о самом тексте
lemmas = db.lemmas # инфа о лемме и токене
dictionary = db.dictionary
plots = db.plots # доп разметка


app = Flask(__name__)


def preprocess(token):
    token = morph.parse(token.strip())[0].normal_form
    return token

def preprocess_sent(sent):
    sent = tokenize(sent)
    res = []
    for start, stop, token in sent:
        res.append(preprocess(token))
    return res




def get_docs(id_docs):
     res = []
     for doc in text_collection.find({'_id': {'$in': id_docs}}):
         ids = [r['_id'] for r in res if r]
         if doc['_id'] not in ids:
             res.append(doc)
     return res


def add_catecory_to_query(category_list, tag_specific):
    sub_query_and = {'$and': []}
    for category in category_list:
        sub_query_or = {'$or': []}
        for tag_cat in category:
            sub_query_or['$or'].append({tag_specific: {'$regex': tag_cat}})
        if sub_query_or['$or'] != []:
            sub_query_and['$and'].append(sub_query_or)
    return sub_query_and


def get_doc_ids(exact: str,
                lemma: str, 
             response_tags: list, 
             response_hero: list, 
             response_plot: list, 
            response_genre='',
            response_info=''):
    
    res_docs = []
    
    query = []
    if exact != '':
        query.append({'token':exact})
    if lemma != '':
        query.append({'iniForm':lemma})

    # запрос для морфологии     
    if response_tags != [[], [], [], [], []]:
        query.append(add_catecory_to_query(response_tags, 'tags'))
    
    
    if query:
        for x in dictionary.find({'$and': query}):
            res_docs+= x['docs']
    
    
    query_plot = []
    # запрос для героев
    if response_hero != [[], [], [], [], [], []]:
        query_plot += add_catecory_to_query(response_hero, 'hero_type')['$and']
        
    #запрос для сюжетов
    if response_plot != [[], [], []]:
        query_plot += add_catecory_to_query(response_plot, 'label')['$and']

    
    # запрос для инфо и жанра     
    query_info_genre = []
    if response_info != '':
        query_info_genre.append({'info' : {'$regex': response_info}})
        
    if response_genre != '':
        query_info_genre.append({'genre' : {'$regex': response_genre}})
    
    res_docs_plot = []
    if query_info_genre != []:
        for x in text_collection.find({'$and': query_info_genre}):
            res_docs_plot.append(x['_id'])
    
        
    result_query = []
    if query != []:
        result_query.append({'id_doc': {'$in': res_docs}})
    if query_plot != []:
        result_query.append({'$and': query_plot})
    if res_docs_plot != []:
        result_query.append({'id_doc': {'$in': res_docs_plot}})
                
    DOCS_FINAL = []        
                
    if result_query != []:
        pipeline_query =  [{'$match' : {'$and': result_query}}, {'$group': {'_id': '$id_doc'}}]
        for x in plots.aggregate(pipeline_query):
            for y in text_collection.find({"_id": x['_id']}):
                DOCS_FINAL.append(y['_id'])

                
    return DOCS_FINAL



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def search():
    print('here')
    a = 'result'
    result = {}
    response_tags = []

    lemma = request.form.get('lemma')
    exact = request.form.get('exact')
    info = request.form.get('info')
    genre = request.form.get('genre')
    response_tags, response_plot, response_hero = [], [], []
    
    for gram in ['POS', 'case', 'number', 'gender',  'other']:
        response_tags.append(request.form.getlist(gram))
    for label in ['b', 's', 'res']:
        response_plot.append(request.form.getlist(label))
    for hero in ['mainhero', 'giver', 'lover', 'helper', 'equal', 'enemy']:
        response_hero.append(request.form.getlist(hero))
    
    result = get_docs(get_doc_ids(exact, lemma, response_tags, response_hero, response_plot, genre, info))

    n_res  = len(result)
    return render_template('result.html', result=result, n_res=n_res)


if __name__ == '__main__':
    app.run(debug=False)
