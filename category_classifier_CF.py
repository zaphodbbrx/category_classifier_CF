#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:48:59 2018

@author: lsm
"""
import pickle
from nlp_utils.TextNormalizer import TextNormalizer
import numpy as np

class category_classifier_CF():
    
    def __init__(self):
        self.lin_model_root = pickle.load(open('pickles/lin_model_root.pkl','rb'))
        self.tn = TextNormalizer().fit()
        
        
    def eval_linmodels(self, message):
        cleaned = self.tn.transform([message])
        root = {
            'decision':self.lin_model_root.predict(cleaned)[0],
            'confidence': np.max(self.lin_model_root.predict_proba(cleaned))
        }
        return root
    
    
    
if __name__ == '__main__':
    cc = category_classifier_CF()
    test_message = '_'
    while test_message!='q':
        test_message = input()
        res = cc.eval_linmodels(test_message)
        print('root category: %s (confidence: %4f)'%(res['decision'],res['confidence']))