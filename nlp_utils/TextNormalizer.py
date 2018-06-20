#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import pymystem3
from nltk.tokenize import TreebankWordTokenizer
import re
from nltk.metrics import edit_distance
import enchant

class SpellingReplacer(object):
    def __init__(self, dict_name = 'ru_RU', max_dist = 2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = 2

    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        suggestions = self.spell_dict.suggest(word)

        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word

class TextNormalizer(BaseEstimator,TransformerMixin):
    def __init__(self, vocab_filename = None):
        self.__tokenizer = TreebankWordTokenizer()
        self.__mystem = pymystem3.Mystem()
        


    
    def spell_check(self, word_list):
        checked_list = []
        for item in word_list:
            replacer = SpellingReplacer()
            r = replacer.replace(item)
            checked_list.append(r)
        return checked_list
    
    
    def __clean_comment(self, text):
        text = str(text)
        if len(text)>0:
            text = re.sub('\W|\d',' ',text).lower()
            tokens = self.__tokenizer.tokenize(text)
            tokens = self.spell_check(tokens)
            tokens = [self.__mystem.lemmatize(t)[0] for t in tokens]
            return ' '.join(tokens)
    
    def transform(self, X, y=None, **fit_params):
        res = []
        for line in X:
            res.append(self.__clean_comment(line))
        return res

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X=None, y=None, **fit_params):
        return self
