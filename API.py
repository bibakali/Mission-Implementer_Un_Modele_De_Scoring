# ------------ Libraries import ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import requests
import pickle
import sklearn
from PIL import Image
import flask
from flask import Flask, render_template,jsonify, request
import config
import zipfile
import json



# Model
import lightgbm as lgb
from lightgbm import LGBMClassifier

import os


# Chargement du meilleur modèle
path_to_best_model = 'best_model.pickle'
with open(path_to_best_model, 'rb') as df_best_model:
    model = pd.read_pickle(df_best_model)

# Chargement du test
path_to_test = 'test_set.pickle'
with open(path_to_test, 'rb') as df_test_appl1:
    test = pd.read_pickle(df_test_appl1)


# Chargement de toutes les informations du client
path_to_test_clean = 'application_test_clean.pickle'
with open(path_to_test_clean, 'rb') as df_test_appl2:
    df_test_clean = pd.read_pickle(df_test_appl2)

# Importer le jeu de données (normale, normalisée) and modele

path1 = df_test_clean
path2 = test

# 1200 Semples 1200 pour df_test
#pd.read_csv(path1, compression='zip')
df_test = path1
df_test = df_test.sample(1200, random_state=42)
df_test = df_test.loc[:, ~df_test.columns.str.match ('Unnamed')]
df_test = df_test.sort_values ('SK_ID_CURR')
# 1200 Semples 1200 pour df_test_normalize
#pd.read_csv(path2, index_col=0)
df_test_normalize = path2
df_test_normalize = df_test_normalize.sample(1200, random_state=42)
df_test_normalize = df_test_normalize.set_index('SK_ID_CURR')
features = df_test_normalize.columns[: -1]
ID_de_client = df_test_normalize.index.sort_values()

app= Flask(__name__, template_folder='templates')

app.config.from_object('config')

# Define the threshold of for application.
threshold = 0.5

# defining flask pages
app = flask.Flask (__name__)
app.config["DEBUG"] = True


# defining home page
@app.route ('/', methods=['GET'])
def home():
    return "<h1>Implémenter Un Modéle de Scoring :</h1><h2> API OK !.</p>"


# defining page for the results of a prediction via index
@app.route ('/scores', methods=['GET'])
def predict():
    # get the index from a request, defined a data_index parameter as default

    if type (flask.request.args.get ('index')) is None:
        data_index = '100038'
    else:
        data_index = flask.request.args.get ('index')

    # get inputs features from the data with index
    df_client = df_test_normalize[df_test_normalize.index == int (data_index)]
    #df_client = df_client.set_index('SK_ID_CURR')
    data = df_client.to_json ()

    # predict_proba returns a list as [0,1], 0 -> for payments accepted, 1 -> for payments refused
    # we have chosen second parameter for refused value
    # Extraction des valeurs du DataFrame Pandas
    input_data = df_client.values
    score = model.predict_proba(input_data)[:,1]
    df_normalize = df_test_normalize.copy ()
    # for add probabilities, used normalized dataset

    df_proba = pd.DataFrame (model.predict_proba(df_normalize)[:, 1], columns=['proba'],
                             index=df_normalize.index.tolist ())
    # for add prediction, used threshold value
    df_proba['Predict'] = np.where (df_proba['proba'] < threshold, 0, 1)

    df_normalize['Proba_Score'] = df_proba['proba']
    df_normalize['Predict'] = df_proba['Predict']

    #  JSON format!
    df_test_new_normalize = df_normalize.to_json ()
    dict_result = { 'Credit_score': score[0], "json_data": data, 'Total_score': df_test_new_normalize }

    # for json format some values are categorical, however it is difficult to handle these values as float,
    # these values types are changed by using JSON encoder
    class NumpyFloatValuesEncoder (json.JSONEncoder):
        def default(self, obj):
            if isinstance (obj, np.float32):
                return float (obj)
            return json.JSONEncoder.default (self, obj)

    # JSON format dumps method for send the data to Dashboard
    dict_result = json.dumps (dict_result, cls=NumpyFloatValuesEncoder)

    # Each request of dashboard, df_normalize dataframe adding ['Proba_Score', 'Predict'] columns,
    # so It needs to drop these columns at the end of the API
    df_normalize.drop (['Proba_Score', 'Predict'], axis=1)

    return dict_result

# define endpoint for Flask

#, 'scores', predict
app.add_url_rule ('/scores', 'scores', predict)
if __name__ == "__main__":
    #os.environ.pop("FLASK_RUN_FROM_CLI")
    app.run(debug=False, port = 5007, use_reloader=False)
    #(host='10.0.5.178', port=6000, debug = True)