# tester app to exercised Flask

from flask import Flask, render_template, request
from string import Template
from OpenSSL import SSL
import pickle
import requests
import json
import pandas as pd
import numpy as np
import time
#model libraries
from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label
from pickle import load
from custom_classes import encode_categorical
from custom_classes import prep_for_keras_input
from custom_classes import fill_empty
from custom_classes import encode_text
from datetime import date
from datetime import datetime
import os
import logging
import yaml

# load config gile
current_path = os.getcwd()
print("current directory is: "+current_path)
path_to_yaml = os.path.join(current_path, 'deploy_web_config.yml')
print("path_to_yaml "+path_to_yaml)
try:
    with open (path_to_yaml, 'r') as c_file:
        config = yaml.safe_load(c_file)
except Exception as e:
    print('Error reading the config file')

# paths for model and pipeline files
pipeline1_filename = config['file_names']['pipeline1_filename']
pipeline2_filename =  config['file_names']['pipeline2_filename']
model_filename =  config['file_names']['model_filename']


# other parms
debug_on = config['general']['debug_on']
logging_level = config['general']['logging_level']
BATCH_SIZE = config['general']['BATCH_SIZE']
score_parameters_from_website = config['general']['score_parameters_from_website']
scoring_columns = config['general']['scoring_columns']

# set logging level
logging_level_set = logging.WARNING
if logging_level == 'WARNING':
    logging_level_set = logging.WARNING
if logging_level == 'ERROR':
    logging_level_set = logging.ERROR
if logging_level == 'DEBUG':
    logging_level_set = logging.DEBUG
if logging_level == 'INFO':
    logging_level_set = logging.INFO   
logging.getLogger().setLevel(logging_level_set)
logging.warning("logging check - beginning of logging")

def get_path(subpath):
    rawpath = os.getcwd()
    # data is in a directory called "data" that is a sibling to the directory containing the notebook
    path = os.path.abspath(os.path.join(rawpath, '..', subpath))
    return(path)

# get complete paths for pipelines and Keras models
'''
pipeline_path = get_path('pipelines')

pipeline1_path = os.path.join(pipeline_path,pipeline1_filename)
pipeline2_path = os.path.join(pipeline_path,pipeline2_filename)
model_path = os.path.join(get_path('models'),model_filename)
'''


# brute force a scoring sample, bagged from test set
score_sample = {}
score_sample['neighbourhood_group'] = 'Brooklyn'
score_sample['neighbourhood'] = 'Park Slope'
score_sample['room_type'] = 'Private room'
score_sample['minimum_nights'] = 14
score_sample['number_of_reviews'] = 35
score_sample['reviews_per_month'] = 0.35
score_sample['calculated_host_listings_count'] = 1





app = Flask(__name__)

HTML_TEMPLATE = Template("""
<h1>Hello ${file_name}!</h1>

<img src="https://image.tmdb.org/t/p/w342/${file_name}" alt="poster for ${file_name}">

""")


@app.route('/')
def home():   
    ''' render home page that is served at localhost and allows the user to enter details about their streetcar trip'''

 
    
@app.route('/show-prediction/')
def show_prediction():
    ''' get the scoring parameters entered in home.html, assemble them into a dataframe, run that dataframe through pipelines
        apply the trained model to the output of the pipeline, and display the interpreted score in show-prediction.html
    '''


 



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    
