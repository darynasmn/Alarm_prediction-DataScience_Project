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


# OUTPUT_FOLDER = "/Users/bober/Desktop/study/naukma/prog/SHIIIIIIIIIIIIIIIIT"
# INPUT_DATA_FOLDER = '/Users/bober/Desktop/study/naukma/prog/SHIIIIIIIIIIIIIIIIT'
# OUTPUT_FOLDER_CSV = '/Users/bober/Desktop/study/naukma/prog/info_for_prediction'
# OUTPUT_DATA_FILE_CSV = '0_predicted_isw'
# INPUT_FOLDER_ML = '/Users/bober/Desktop/study/naukma/prog/models'
# BASA_URL = "https://understandingwar.org/backgrounder/russian-offensive-campaign-assessment"
# OUTPUT_FOLDER_PREDICTION = "/Users/bober/Desktop/study/naukma/prog/predictions"

OUTPUT_FOLDER = "save"
INPUT_DATA_FOLDER = 'save'
OUTPUT_FOLDER_CSV = 'info_for_prediction'
OUTPUT_DATA_FILE_CSV = '0_predicted_isw'
INPUT_FOLDER_ML = 'models'
BASA_URL = "https://understandingwar.org/backgrounder/russian-offensive-campaign-assessment"
OUTPUT_FOLDER_PREDICTION = "predictions"


tf_idf_model = pickle.load(open ("td/tfidf_transformer_v1.pkl", "rb"))
cv_model = pickle.load(open ("td/count_vectorizer_v1.pkl", "rb"))




# app = Flask(__name__)
# class InvalidUsage(Exception):
#     status_code = 400

#     def __init__(self, message, status_code=None, payload=None):
#         Exception.__init__(self)
#         self.message = message
#         if status_code is not None:
#             self.status_code = status_code
#         self.payload = payload

#     def to_dict(self):
#         rv = dict(self.payload or ())
#         rv["message"] = self.message
#         return rv

# @app.errorhandler(InvalidUsage)
# def handle_invalid_usage(error):
#     response = jsonify(error.to_dict())
#     response.status_code = error.status_code
#     return response



def save_page(url, file_name):
    temp_url = url
    temp_url = temp_url.split('-') 
    if temp_url[-2] < '10':
        url = url[:len(url)-5]
        if 'january' in url:
            url = url.replace('0','') + '-2023'
        else:
            url = url.replace('0','')
    page = requests.get(url)
    url_name = url.split("/")[-1].replace("-", "_")
    with open(f"{OUTPUT_FOLDER}/{file_name}__{url_name}.html", 'wb+') as f:
        f.write(page.content)

def get_prev_date(what):
    mnths = ["january", "february", "march", "april", "may", "june", "july", "august","september", "october", 
             "november", "december"]
    base = datetime.datetime.strptime(datetime.datetime.now(pytz.timezone('Europe/Kyiv')).strftime("%Y-%m-%d"), "%Y-%m-%d") - timedelta(days=1)
    if what == "m":
        return mnths[int(base.strftime("%m"))-1]
    elif what == "d":
        return base.strftime("%d")
    elif what == "year":
        return base.strftime("%Y")
    elif what == "all":
        return base
    
    

def download_open_file():
    month = get_prev_date("m")
    day = get_prev_date("d")
    year = get_prev_date("year")
    url = f"{BASA_URL}-{month}-{day}-{year}"
    file_name = f"{month}_{day}_{year}"


    file_name_2 =  file_name+'__russian_offensive_campaign_assessment_'+file_name+'.html'

    if not os.path.isfile(os.path.join(OUTPUT_FOLDER, file_name_2)):
        save_page(url, file_name)


def get_isw_report():
    #1
    download_open_file()
    #2
    month = get_prev_date("m")
    day = get_prev_date("d")
    year = get_prev_date("year")
    url = f"{BASA_URL}-{month}-{day}-{year}"
    file_name_x = f"{month}_{day}_{year}"
    to_open = file_name_x+'__russian_offensive_campaign_assessment_'+file_name_x+'.html'
    i = glob.glob(f'{OUTPUT_FOLDER}/{to_open}')[0]
    all_data = []
    d={}
    file_name = i.split('/')[-1].split('__') #here
    date = datetime.datetime.strptime(file_name[0], '%B_%d_%Y')
    url = file_name[1].split('.')[0]
    with open(i, 'r', encoding="utf-8", errors='namereplace') as cfile:
                parsed_html = BeautifulSoup(cfile, 'html.parser')
                try:
                    title = parsed_html.head.find('title').text
                except AttributeError:
                    title = ""
                try:
                    link = parsed_html.head.find('link', attrs={'rel':"canonical"}, href = True).attrs["href"]
                except (AttributeError, KeyError):
                    link = ""
                try:
                    text_title = parsed_html.body.find('h1', attrs={'id':'page-title'}).text
                except AttributeError:
                    text_title = ""
                try:
                    text_main = parsed_html.body.find('div', attrs={'class':'field field-name-body field-type-text-with-summary field-label-hidden'}).decode_contents(formatter="html")
                except AttributeError:
                    text_main = ""

                dictionary = {
                    'date':date,
                    'short_url':url,
                    'title':title,
                    'text_title':text_title,
                    'full_url':link,
                    'main_html':text_main
                }

                all_data.append(dictionary)
    df = pd.DataFrame.from_dict(all_data)
    df['main_html_v2'] = df['main_html'].apply(lambda x: UTILS.remove_names_and_dates(x, BeautifulSoup))
    pattern = '\[(\d+)\]'

    df['main_html_v3'] = df['main_html_v2'].apply(lambda x: re.sub(pattern, '', x))
    df['main_html_v4'] = df['main_html_v3'].apply(lambda x: BeautifulSoup(x).text)
    df['main_html_v5'] = df['main_html_v4'].apply(lambda x: re.sub(r'http(\S+.*\s)','',x))
    df['main_html_v6'] = df['main_html_v5'].apply(lambda x: re.sub(r'(o2022|o2023|2022|2023)','',x))
    df['main_html_v7'] = df['main_html_v6']
    def del_clickHere(df):
        divider = "in this report."
        for i in range(df.shape[0]):
            if divider in df.loc[i,'main_html_v6']:
                temp_str = df.loc[i,'main_html_v6']
                temp_list = temp_str.split(divider)
                df.loc[i,'main_html_v7'] = temp_list[1]
    del_clickHere(df)
    def del_clickHere(df):
        divider = "invasion of Ukraine."
        for i in range(df.shape[0]):
            if divider in df.loc[i,'main_html_v7']:
                temp_str = df.loc[i,'main_html_v7']
                temp_list = temp_str.split(divider)
                df.loc[i,'main_html_v7'] = temp_list[1]
    del_clickHere(df)
    df['main_html'] = df['main_html'].apply(lambda x: BeautifulSoup(x).text)
    df2 = df.drop(['main_html_v2','main_html_v3','main_html_v4',
                   'main_html_v5', 'main_html_v6'], axis = 1)
    def preprocess(data, word_root_algo= "lemm"):
        from UTILS import remove_one_letter_word,remove_url_string, convert_lower_case, remove_punctuation, remove_apostrophe, remove_stop_words, conver_numbers, stemming, lemmatizing
        data = UTILS.remove_one_letter_word(data)
        data = UTILS.remove_url_string(data)
        data = UTILS.convert_lower_case(data)
        data = UTILS.remove_punctuation(data)
        data = UTILS.remove_apostrophe(data)
        data = UTILS.remove_stop_words(data)
        data = UTILS.conver_numbers(data)
        data = UTILS.stemming(data)
        data = UTILS.remove_punctuation(data)
        data = UTILS.conver_numbers(data)

        if word_root_algo == "lemm":
            data = UTILS.lemmatizing(data)
        else:
            data = UTILS.stemming(data)

        data = UTILS.remove_punctuation(data)
        data = UTILS.remove_stop_words(data)

        return data
    df2['report_text_lemm']= df2['main_html_v7'].apply(lambda x: preprocess(x,'lemm'))
    df2.to_csv(f'{OUTPUT_FOLDER_CSV}/{OUTPUT_DATA_FILE_CSV}', sep =';', index = False)
    docs = df2['report_text_lemm'].tolist()
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(docs)
    with open('models/pred_count_vectorizer_v1.pkl', 'wb') as handle:
        pickle.dump(cv ,handle)
    tfidf_transformer =  TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    with open('models/pred_tfidf_transformer_v1.pkl', 'wb') as handle:
        pickle.dump(tfidf_transformer, handle)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index = cv.get_feature_names_out(), columns = ['idf_weights'])
    tf_idf_vector = tfidf_transformer.transform(word_count_vector)
    tfidf = pickle.load(open('models/pred_tfidf_transformer_v1.pkl', 'rb'))
    cv = pickle.load(open('models/pred_count_vectorizer_v1.pkl', 'rb'))
    def conver_doc_to_vector(doc):
        feature_names = cv.get_feature_names_out()
        top_n = 100
        tf_idf_vector = tfidf.transform(cv.transform([doc]))

        sorted_items = tf.sort_coo(tf_idf_vector.tocoo())

        keywords = tf.extract_topn_from_vector(feature_names, sorted_items,top_n)

        return keywords
    from utils import tf
    df2['keywords'] = df2['report_text_lemm'].apply(lambda x: conver_doc_to_vector(x))
    df2['report_date']=df2['date']
    df2['date_tmrw']= (pd.to_datetime(df2['date'])+datetime.timedelta(days=1)).dt.strftime('%Y-%m-%d')
    isw_df = df2[["report_date", "date_tmrw", "keywords", "main_html_v7", "report_text_lemm"]].copy().add_prefix('isw_')
    return isw_df

def save_file(data,city, date):
    data_object = json.dumps(data)

    # open file for writing, "w" 
    f = open(f"{SAVED_FORCASTS}/{city}_{date}.json","w")

    # write json object to file
    f.write(data_object)

    # close file
    f.close()


def read_file(path):
    f = open(path)
  
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
  
  
    # Closing file
    f.close()
    return data

def get_weather(city, date):
    
    path = f"{SAVED_FORCASTS}/{city}_{date}.json"
    if (os.path.exists(path)):
        jsonData = read_file(path)
        return jsonData
    location = f"{city},Ukraine"
    url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date}?key={API_KEY}&include=hours&unitGroup=metric&contentType=json'
    try: 
      ResultBytes = urllib.request.urlopen(url)
  
      # Parse the results as JSON
      jsonData = json.load(ResultBytes)

        
    except urllib.error.HTTPError  as e:
      ErrorInfo= e.read().decode() 
      print('Error code: ', e.code, ErrorInfo)
      sys.exit()
    except  urllib.error.URLError as e:
      ErrorInfo= e.read().decode() 
      print('Error code: ', e.code,ErrorInfo)
      sys.exit()
    save_file(jsonData,city, date)
    return jsonData



def get_next_date(date):
    return (date+datetime.timedelta(days=1)).strftime("%Y-%m-%d")


def get_df_weather(jsonData):
    df_data_day = pd.DataFrame(jsonData['days'])
    df_data_day = df_data_day[df_data_day.columns[0:33]].add_prefix('day_')
    hours_forecast=jsonData['days'][0]['hours']
    df_weather_hours = pd.DataFrame(hours_forecast).add_prefix('hour_')
    df_weather_hours['hour_int']=pd.to_datetime(df_weather_hours['hour_datetime']).dt.hour
    df_weather_hours['key'] = 1
    df_data_day['key'] = 1
    df_weather_final = pd.merge(df_data_day,df_weather_hours, on='key')
    return df_weather_final


def get_weather_for_12_hours(city,date):
    
    df_regions = pd.read_csv(DIR_REGIONS)
    jsonData = get_weather(city, date.strftime("%Y-%m-%d"))
    current_hour = int(date.strftime("%H"))
    weather_all_data_day1 = get_df_weather(jsonData)
    hours_needed = (weather_all_data_day1['hour_int']>=current_hour)&(weather_all_data_day1['hour_int']<=(current_hour+12))
    weather_all_data_day1=weather_all_data_day1[hours_needed]
    df_weather_final = weather_all_data_day1
    hours_left=12-weather_all_data_day1.shape[0]
    if(hours_left>0):
        jsonData = get_weather(city, get_next_date(date))
        weather_all_data_day2 = get_df_weather(jsonData)
        hours_needed_2 = ((weather_all_data_day2['hour_int']<=hours_left))
        weather_all_data_day2=weather_all_data_day2[hours_needed_2]
        df_weather_final = pd.concat([weather_all_data_day1, weather_all_data_day2], axis=0)
    df_weather_final['city']=city
    df_final = pd.merge(df_weather_final,df_regions,left_on="city",right_on="center_city_en")


    return df_final


def save_prediction(prediction):
    data = json.dumps(prediction)

    # open file for writing, "w" 
    f = open(f"{OUTPUT_FOLDER_PREDICTION}/prediction.json","w")

    # write json object to file
    f.write(data)

    # close file
    f.close()

def save_prediction_with_time(prediction):
    data = json.dumps(prediction)
    time_now = datetime.datetime.now(pytz.timezone('Europe/Kyiv')).strftime('%m/%d/%yT%H:%M:%S').replace(':','_').replace('/','_')
    # open file for writing, "w" 
    data = json.dumps(prediction)

    # open file for writing, "w" 
    f = open(f"{OUTPUT_FOLDER_PREDICTION}/prediction_{time_now}.json","w")

    # write json object to file
    f.write(data)

    # close file
    f.close()
    
def get_prediction(path):
    return read_file(path)
    
    
def new_prediction():
    name = 'logistic_regression'
    version = 'v1'
    name_model = f'6__{name}__{version}'
    model_4 = pickle.load(open (f"{INPUT_FOLDER_ML}/{name_model}.pkl", "rb"))
    isw_df = get_isw_report()
    cities = ['Vinnytsia','Simferopol','Lutsk','Dnipro','Donetsk','Zhytomyr','Uzhgorod','Zaporozhye','Ivano-Frankivsk','Kyiv','Kropyvnytskyi',
             'Luhansk','Lviv','Mykolaiv','Odesa','Poltava','Rivne','Sumy','Ternopil','Kharkiv','Kherson','Khmelnytskyi',
             'Cherkasy','Chernivtsi','Chernihiv']
    date = datetime.datetime.now(pytz.timezone('Europe/Kyiv'))
    result = {}
    for city in cities:
        df_weather_final = get_weather_for_12_hours(city,date)
        df_weather_final['key']=1
        isw_df['key']=1
        df_all = df_weather_final.merge(isw_df, how = 'left', left_on = 'key', right_on = 'key')
        to_drop=['key','isw_report_date','isw_date_tmrw','isw_keywords','isw_main_html_v7','isw_report_text_lemm']
        df_weather_matrix_v1 = df_all.drop(to_drop, axis = 1)
        df_weather_matrix_v1= df_weather_matrix_v1[['day_tempmax', 'day_tempmin', 'day_temp', 'day_dew', 'day_humidity',
               'day_precip', 'day_precipcover', 'day_solarradiation',
               'day_solarenergy', 'day_uvindex', 'hour_temp', 'hour_humidity',
               'hour_dew', 'hour_precip', 'hour_precipprob', 'hour_snow',
               'hour_snowdepth', 'hour_windgust', 'hour_windspeed', 'hour_winddir',
               'hour_pressure', 'hour_visibility', 'hour_cloudcover',
               'hour_solarradiation', 'hour_uvindex', 'hour_severerisk','region_id']]
        cv_vector_model = cv_model.transform(df_all['isw_report_text_lemm'].values.astype('U'))
        TF_IDF_MODEL = tf_idf_model.transform(cv_vector_model)
        df_weather_matrix_v1_csr = scipy.sparse.csr_matrix(df_weather_matrix_v1.values)
        df_all_data_csr = scipy.sparse.hstack((df_weather_matrix_v1_csr, TF_IDF_MODEL), format='csr')
        predicted = model_4.predict(df_all_data_csr)
        current_time = datetime.datetime.now(pytz.timezone('Europe/Kyiv'))
        hours = []
        result['time_of_prediction'] = current_time.strftime('%m/%d/%y %H:%M:%S')
        predicted_str = []
        for m in predicted:
            predicted_str.append(str(m))
        for i in range(13):
            hour = date + datetime.timedelta(hours=i)
            hour_rounded = hour.replace(minute=0, second=0, microsecond=0)
            hours.append(hour_rounded.strftime('%H:%M'))

        result[city] = dict(zip(hours, predicted_str))
        save_prediction(result)
        save_prediction_with_time(result)
    return result

# @app.route("/check")
# def home_page():
#     return "<p><h2>KMA L2: Python Saas.</h2></p>"

# @app.route(
#     "/new_prediction",
#     methods=["POST"],
# )
def get_new_prediction():
    result = new_prediction()
    return result

# @app.route(
#     "/get_prediction",
#     methods=["POST"],
# )

# def prediction_endpoint():
#     querstring = request.get_json()

#     if querstring.get("token") is None:
#         raise InvalidUsage("token is required", status_code=400)

#     token = querstring.get("token")

#     if token != API_TOKEN:
#         raise InvalidUsage("wrong API token", status_code=403)
    
#     path = f"{OUTPUT_FOLDER_PREDICTION}/prediction.json"
#     prediction = get_prediction(path)
#     prediction_all_regions = prediction
#     time_prediction = datetime.datetime.strptime(prediction['time_of_prediction'],'%m/%d/%y %H:%M:%S')
#     del prediction_all_regions['time_of_prediction']
#     if ((querstring.get("region")=="all")|(querstring.get("region")=="")):
#         data_to_show = prediction_all_regions
#     else:
#         data_to_show = prediction_all_regions[querstring.get("region")]
        
#     result = {
#         "last_model_train_time": '12-04-2023, 16:50:11',
#         "last_prediciotn_time": time_prediction.isoformat(),
#         "region": querstring.get("region"),
#         "regions_forecast": data_to_show
#     }
#     return result
