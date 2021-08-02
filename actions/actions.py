# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import re

data_df = pd.read_csv("phishing/web_dataset_small.csv")
#feat_top = ["time_domain_activation","directory_length","length_url","qty_slash_directory","ttl_hostname","asn_ip","qty_slash_url","file_length","time_response","time_domain_expiration"]
feat_top = ["directory_length","length_url","qty_slash_directory","ttl_hostname","qty_slash_url","file_length"]
y = data_df["phishing"]
x = data_df[feat_top]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)



def str_length(s):
  return len(s)


def extract_urlfeatures(sample):
  length_dict = {}

  # Extracting features
  domainname =  re.findall('://www.([\w\-\.]+)', sample)[0]
  pathname = "/".join(sample.split("/")[3:])
  directoryname = "/".join(pathname.split("/")[:1])
  filename_params = "/".join(pathname.split("/")[1:])
  filename = "?".join(filename_params.split("?")[:1])
  params_combined = "?".join(filename_params.split("?")[1:])
  params = "".join(re.findall('=([\w\-\.]+)', params_combined))

  length_dict["length_url"] = str_length("www"+domainname)
  length_dict["qty_slash_directory"] = directoryname.count("/")
  length_dict["qty_slash_url"] = sample.count("/")
  length_dict["ttl_hostname"] = 2977
  length_dict["file_length"] = str_length(filename)
  length_dict["directory_length"] = str_length(directoryname)

  df = pd.DataFrame([length_dict])
  return df


class ActionExerciseUrlCheck(Action):

    def name(self) -> Text:
        return "action_exercise_url"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        sample_url = "https://www.webmd.com/fitness-exercise/features/7-most-effective-exercises"

        d_df = extract_urlfeatures(sample_url)
        d_df = scaler.transform(d_df)
        print(rfc.predict(d_df))
        text = "Here is the URL to help you:https://www.webmd.com/fitness-exercise/features/7-most-effective-exercises"
        dispatcher.utter_message(text=text)

        return []

class ActionDietUrlCheck(Action):

    def name(self) -> Text:
        return "action_diet_url"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        sample_url = "https://www.randomwebsite.com/@@123/734568"

        d_df = extract_urlfeatures(sample_url)
        d_df = scaler.transform(d_df)
        print(rfc.predict(d_df))
        text = "Here is the URL to help you: "+sample_url+" (the information url looks suspicious. Please open with caution)"
        dispatcher.utter_message(text=text)

        return []

class ActionStressUrlCheck(Action):

    def name(self) -> Text:
        return "action_stress_url"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        sample_url = "https://www.heart.org/en/healthy-living/healthy-lifestyle/stress-management/what-is-stress-management"

        d_df = extract_urlfeatures(sample_url)
        d_df = scaler.transform(d_df)
        print(rfc.predict(d_df))
        text = "Here is the URL to help you:"+sample_url
        dispatcher.utter_message(text=text)

        return []
