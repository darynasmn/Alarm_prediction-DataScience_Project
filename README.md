# Alarm_prediction-DataScience_Project
 
## Introduction:
This repository contains a Python SaaS solution that predicts the probability of air alarms for all regions in Ukraine every hour for the next 12 hours. The project was developed as a University project under the supervision of Professor Andrew Kurochkin by a team of four consisting of Koval Sviatoslav, Kotliarenko Anastasiia and Semenets Daryna. 

## Endpoint:
The project's endpoint is a Python script called "5_prediction(2).py," which is executed every hour to generate predictions of air alarms for the next 12 hours.

## How to use:
To use the SaaS, clone this repository and run the "5_prediction(2).py" file. The script will generate predictions for the next 12 hours for all regions of Ukraine.With the help of cron, we implemented the ability to model's prediction every hour.


## Overview:
The project uses various data sources, including weather data, ISW reports, past air alarms, and war articles, to predict the probability of air alarms. The project uses a combination of natural language preprocessing, exploratory data analysis, Pandas manipulations, data modeling, and machine learning algorithms to predict the likelihood of air alarms.


## Why:
- Predict and inform people = more time to prepare​

- Authorities have more time to prepare, inform and prevent ​

- Analysis and conclusions leads to protecting those regions who are in great danger​


## Problem statement:
The issue we are facing is the uncertainty around when alarms will arise. Being able to forecast instances of alarms would greatly benefit us in both our daily lives and in planning for the future. Our approach to addressing this issue involves leveraging machine learning to develop a model capable of predicting instances of air alarms.​

## Implementation:
The project was developed using Python for data science, and it includes web scraping and preprocessing techniques. The SaaS is implemented using Amazon EC2 servers, and PostMan collections were created to facilitate API calls.

## Conclusion:
This project is a great example of how data science can be used to develop real-world solutions to everyday problems. The SaaS solution developed here provides valuable information to Ukrainian citizens and helps them make informed decisions about their safety.
