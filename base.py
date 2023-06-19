# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:45:09 2023

@author: Xiaomi
"""
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

client = pymongo.MongoClient('localhost', 27017, unicode_decode_error_handler='ignore')
db = client['mongo_for_folklore']
text_collection = db.texts # инфа о самом тексте
lemmas = db.lemmas # инфа о лемме и токене
plots = db.plots # доп разметка
def preprocess(token):
    token = morph.parse(token.strip())[0].normal_form
    return token

def preprocess_sent(sent):
    sent = tokenize(sent)
    res = []
    for start, stop, token in sent:
        res.append(preprocess(token))
    return res

print(preprocess_sent('Что-то и 5пр и на англ. English English-02'))
# =============================================================================
# for x in lemmas.find({'iniForm': 'что-то'}):
#     for doc in text_collection.find({'_id': {'$in': x['docs']}}):
#         print(doc['doc_text'])
# =============================================================================
