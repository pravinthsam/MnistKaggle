# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 08:12:01 2016

@author: osboxes
"""

from keras.models import model_from_json
import csv

name = 'mnist_simple_DNN_2'
model = model_from_json(open('./models/' + name + '.json').read())
model.load_weights('./models/' + name + '_weights.h5')








