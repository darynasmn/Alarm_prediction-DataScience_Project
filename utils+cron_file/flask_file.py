import all_methods
import pandas as pd
import requests
import time
import glob
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from IPython.core.display import display, HTML
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from num2words import num2words
import nltk
import os
import string
import numpy as np
import copy
import pickle
import re
import math
import datetime
import pytz
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sys
sys.path.append('utils')
import UTILS as U
from UTILS import *
from utils import UTILS
import urllib.request
import sys
import datetime
import json
import pytz
import os
import scipy
import pandas as pd  
import requests
from flask import Flask, jsonify, request

API_KEY = "V8J7UYNGWSQ65E3KGUDZ4YKF4"
API_TOKEN = 'alarm_predict'
DIR_REGIONS = 'raw_data/regions.csv'
SAVED_FORCASTS = 'saved_weather'
OUTPUT_FOLDER = "save"
INPUT_DATA_FOLDER = 'save'
OUTPUT_FOLDER_CSV = 'info_for_prediction'
OUTPUT_DATA_FILE_CSV = '0_predicted_isw'
INPUT_FOLDER_ML = 'models'
BASA_URL = "https://understandingwar.org/backgrounder/russian-offensive-campaign-assessment"
OUTPUT_FOLDER_PREDICTION = "predictions"

tf_idf_model = pickle.load(open ("td/tfidf_transformer_v1.pkl", "rb"))
cv_model = pickle.load(open ("td/count_vectorizer_v1.pkl", "rb"))

app = Flask(__name__)
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route("/check")
def home_page():
    return "<p><h2>KMA L2: Python Saas.</h2></p>"

@app.route(
    "/new_prediction",
    methods=["POST"],
)
def get_new_prediction():
    return all_methods.new_prediction()

@app.route(
    "/get_prediction",
    methods=["POST"],
)

def prediction_endpoint():
    querstring = request.get_json()

    if querstring.get("token") is None:
        raise InvalidUsage("token is required", status_code=400)

    token = querstring.get("token")

    if token != API_TOKEN:
        raise InvalidUsage("wrong API token", status_code=403)
    
    path = f"{OUTPUT_FOLDER_PREDICTION}/prediction.json"
    prediction = get_prediction(path)
    prediction_all_regions = prediction
    time_prediction = datetime.datetime.strptime(prediction['time_of_prediction'],'%m/%d/%y %H:%M:%S')
    del prediction_all_regions['time_of_prediction']
    if ((querstring.get("region")=="all")|(querstring.get("region")=="")):
        data_to_show = prediction_all_regions
    else:
        data_to_show = prediction_all_regions[querstring.get("region")]
        
    result = {
        "last_model_train_time": '12-04-2023, 16:50:11',
        "last_prediciotn_time": time_prediction.isoformat(),
        "region": querstring.get("region"),
        "regions_forecast": data_to_show
    }
    return result
