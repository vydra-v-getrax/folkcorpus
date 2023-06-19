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
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="k")
ner = NER.load('slovnet_ner_news_v1.tar')
navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
ner.navec(navec)

nltk.download("stopwords")
morph = MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

# Загрузка базы
client = pymongo.MongoClient('localhost', 27017, unicode_decode_error_handler='ignore')
db = client['mongo_for_folklore']
text_collection = db.texts # инфа о самом тексте
lemmas = db.lemmas # инфа о лемме и токене
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


def get_by_lemma(lemma: str, response_tags: list):
    """ все кроме леммы идет в forms.tags и подается в списке response_tags """
    res = []
#     lemma = 'старый'
#     response_tags = ['ед']
    list_of_tags = []
    if response_tags:
        list_of_tags = [{'forms.tags': re.compile(i)} for i in response_tags]
    if lemma: 
        list_of_tags.append({'iniForm': preprocess(lemma)})
    if list_of_tags:
        for x in lemmas.find({'$and': list_of_tags}):
            for doc in text_collection.find({'_id': {'$in': x['docs']}}):
                res.append(doc['_id'])
    return res

def get_doc_ids(lemma: str, 
             response_tags: list, 
             response_hero: list, 
             response_plot: list, 
            response_genre='',
            response_info=''):
    """ все кроме леммы идет в forms.tags и подается в списке response_tags """
    res = []
#     lemma = 'старый'
#     response_tags = ['ед']
    list_of_tags = []
    if response_tags:
        list_of_tags = [{'forms.tags': re.compile(i)} for i in response_tags]
    if lemma: 
        list_of_tags.append({'iniForm': preprocess(lemma)})
    if list_of_tags:
        for x in lemmas.find({'$and': list_of_tags}):
            for doc in text_collection.find({'_id': {'$in': x['docs']}}):
                res.append(doc['_id'])
#                 print(doc)
                
    if response_hero:
        hero = []
        for y in plots.aggregate([{'$match' : {'hero_type': {'$in': response_hero}}}, {'$group': {'_id': '$id_doc'}}]):
            hero.append(text_collection.find({'_id': y['_id']})[0]['_id'])
                
    if response_plot:
#         print(response_plot)
        plot = []
        for x in plots.find({'label': {'$in': response_plot}}):
            plot.append(text_collection.find({"_id": x['id_doc']})[0]['_id'])
            
        plot = []
        for x in plots.find({'$and':[{'genre' : {'$regex': response_genre}}, 
                                     {'info': {'$regex': response_info}}]}):
            info.append(text_collection.find({"_id": x['id_doc']})[0]['_id'])
    return res


def get_docs(id_docs):
    res = []
    for doc in text_collection.find({'_id': {'$in': id_docs}}):
        ids = [r['_id'] for r in res if r]
        if doc['_id'] not in ids:
            res.append(doc)
    return res



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def search():
    print('here')
    a = 'result'
    result = {}
    response_tags = []
# =============================================================================
#     if request.args:
#         if 'exact' in request.args:
#             exact = request.args['exact']
#             print('exact', exact)
#         if 'lemma' in request.args:
#             lemma = request.args['lemma']
#             print('lemma', lemma)
#         if 'info' in request.args:
#             info = request.args['info']
#             print('info', info)
#         if 'genre' in request.args:
#             genre = request.args['genre']
#             print('genre', genre)
#         print(request.args)
# =============================================================================
    lemma = request.form.get('lemma')
    exact = request.form.get('exact')
    info = request.form.get('info')
    genre = request.form.get('genre')
    response_tags, response_plot, response_hero = [], [], []
    for gram in ['POS', 'case', 'number', 'gender',  'other']:
        response_tags.extend(request.form.getlist(gram))
    for label in ['b', 's', 'res']:
        response_plot.extend(request.form.getlist(label))
    for hero in ['mainhero', 'giver', 'lover', 'helper', 'equal', 'enemy']:
        response_hero.extend(request.form.getlist(hero))
    
    result = get_docs(get_doc_ids(lemma, response_tags, response_hero, response_plot, genre, info))
    print(type(result[0]['doc_text']))
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=False)
