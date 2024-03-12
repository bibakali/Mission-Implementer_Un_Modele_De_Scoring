# ------------ Libraries import ---------------------------
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import requests
import pickle
import sklearn
from PIL import Image
# Model
import lightgbm as lgb
from lightgbm import LGBMClassifier
import json

# Chargement du meilleur modèle
fic_best_model = 'best_model.pickle'
with open(fic_best_model, 'rb') as df_best_model:
    model = pickle.load(df_best_model)

# Chargement du test
fic_test_appl = 'test_set.pickle'
with open(fic_test_appl, 'rb') as df_test_appl:
    test = pickle.load(df_test_appl)

    
# Chargement de toutes les informations du client
fic_test_appl = 'application_test_clean.pickle'
with open(fic_test_appl, 'rb') as df_test_appl:
    df_test_clean = pickle.load(df_test_appl)

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
df_test_copy = df_test_normalize.copy()
df_test_normalize = df_test_normalize.set_index('SK_ID_CURR')
features = df_test_normalize.columns[: -1]
ID_de_client = df_test_normalize.index.sort_values()




## importer les images ##
img1 = Image.open(r'ent.png')


def load_infos_gen(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    gender = data.CODE_GENDER.value_counts()

    return nb_credits, rev_moy, credits_moy, gender

# ------------ Affichage principal -----------------
## Titre générique ##
# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header="""
    <head>
        <title>Application Dashboard Crédit Score</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>             
    <h1 style="font-size:300%; color:Blue; font-family:Arial"> Prêt à dépenser <br>
        <h2 style="color:#FFBB33; font-family:Georgia"> DASHBOARD</h2>
        <hr style= "  display: block;
          margin-top: 0;
          margin-bottom: 0;
          margin-left: auto;
          margin-right: auto;
          border-style: inset;
          border-width: 1.5px;"/>
     </h1>
"""
st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="logo", layout="wide")
st.markdown(html_header, unsafe_allow_html=True)

# ------------ Configuration de la barre latérale (sidebar) ----------------------


### Afficher l'image avec streamlit ###
st.sidebar.image(img1)

### Ajouter une colonne pour input de l'utilisateur ###
st.sidebar.header('Definir ID de client:')
selected_credit = st.sidebar.selectbox('ID de client', ID_de_client)
### Ajouter checkbox pour afficher différentes informations client ###
client_data = st.sidebar.checkbox('Informations générales')
client_pred_score = st.sidebar.checkbox('Analyse de la demande de crédit')

# Chargement des informations générales
st.sidebar.header("**Informations générales**")
nb_credits, rev_moy, credits_moy, gender = load_infos_gen(df_test)

# Revenu moyen
st.sidebar.markdown("<u>Revenu moyen (USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

# AMT CREDIT
st.sidebar.markdown("<u>Montant moyen du prêt (USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(credits_moy)



## Afficher input dataframe avec une sélection multiple de fonctionnalités pour toute la liste de passagers disponible (les données ne sont pas mises à l'échelle standard ici !) ##
html_select_client="""
    <div class="card">
      <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: black; padding-top: 5px; width: auto;
                  height: 40px;">
        <h3 class="card-title" style="background-color:#FFBB33; color:black;
                   font-family:Georgia; text-align: center; padding: 0px 0;">
          Informations sur les clients
        </h3>
      </div>
    </div>
    """

st.markdown(html_select_client, unsafe_allow_html=True)
selections = st.multiselect('Vous pouvez ajouter ou enlever une donnée présente dans cette liste:', df_test_normalize.columns.tolist(),
df_test_normalize.columns.tolist()[0:10])
st.dataframe(df_test_normalize.loc[:,selections].sort_index())
### ajouter un extenseur pour plus d'explications sur les données  ###
with st.expander('Informations complémentaires'):
    st.write(""" Ici vous trouvez les informations disponibles pour tous les clients.  \n"""
            """ Pour plus d'informations sur les features (variables) disponibles merci de contacter l'équipe support. """)



colors = ['red', 'green']

fig = go.Figure(data=[go.Pie(labels=[ "Default", "Non Default"],
                             values=[8.3 , 91.7])])
fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=20,pull=[0, 0.1],
                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))

st.write("### Proportion de 'TARGET'")
st.plotly_chart(fig)




colors = ['blue', 'pink']
fig = go.Figure(data=[go.Pie(labels=[ "Hommes", "Femmes"],
                             values=[65 , 35])])
fig.update_traces(hoverinfo='label+percent', textinfo='label+percent', textfont_size=20,pull=[0, 0.05],
                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))
st.write("### Proportion de 'GENDRE'")
st.plotly_chart(fig)









## Afficher les données client sélectionnées (condition de checkbox : 'Données client') ##
if client_data:
    st.write(f'### Données du client, demande {selected_credit}')
    ### définir les valeurs à afficher pour l'histogramme et les données client (avec un maximum à 5) ###
    selections_client0 = st.multiselect('Vous pouvez afficher 5 données maximum parmi cette liste:', df_test_normalize[features].columns.tolist(),
    df_test_normalize[features].columns.tolist()[1:5])
    ### définir des colonnes pour diviser un visuel en deux  ###
    col1, col2 = st.columns(2)
    ### Afficher les informations client concernant les features sélectionnées ###
    col1.dataframe(df_test_normalize.loc[selected_credit, selections_client0])
    ### définir pyplot pour col2 barchart avec les informations sur les passagers sélectionnés avec condition du nombre de features sélectionnées ###
    
    if len(selections_client0) <= 5:
        dict_tmp = {'Features': df_test_normalize[features].loc[selected_credit, selections_client0].index,
                    'Valeur normalisée': df_test_normalize[features].loc[selected_credit, selections_client0].values}
 
        df_temp = pd.DataFrame(dict_tmp)
        fig = px.bar(        
           df_temp,
           x = "Features",
           y = "Valeur normalisée",
           title = f'Diagramme bar données ID: {selected_credit}',hover_data= ['Features', 'Valeur normalisée'] 
          )
        st.plotly_chart(fig)

    else:
        col2.write("Vous avez sélectionné trop de feature!!! Le graphique n'a pas pu être affiché")

    ### ajouter un extenseur pour plus d'explications sur les données client sélectionnées  ###
    with st.expander('Informations complémentaires'):
        st.write(""" Ici vous trouvez les informations client disponibles pour la demande de prêt sélectionnée.  \n"""
            """ La graphique en bâton donne les valeurs de features (variables) normalisées pour pouvoir les afficher sur la même échelle. """)





## Afficher la réponse du prêt concernant le calcul de probabilité du modèle ( via API Flask pour obtenir le résultat / checbox condition : 'Résultat de la demande de prêt') ##
if client_pred_score:
    st.write('### Décision sur la demande de prêt')
    ###  attention l'url de l'API doit être changée pour le déploiement en série !!###
    ####https://bbkalilunix.pythonanywhere.com/scores
    url_api_model_result = 'https://bbkalilunix.pythonanywhere.com/scores'
    ### Faites attention aux paramètres, avec doit avoir un dict avec la valeur de prêt d'index / ID. C'est ainsi qu'il est implémenté dans notre API ###
    get_request = requests.get(url=url_api_model_result, params={'index': selected_credit})
    if get_request.status_code == 200:
        data = get_request.json()
        if 'Credit_score' in data:
            score_client = data['Credit_score']
        else:
            st.write("La réponse ne contient pas de clé 'Credit_score'")
    else:
        st.write("La requête GET a échoué avec le code de statut :", get_request.status_code)
    ### Nous obtenons les informations de prédiction à partir du format json du modèle d'API  ###
    score_client = get_request.json()['Credit_score']
    ### Nous obtenons la réponse concernant l'acceptation du prêt ######
    if get_request.status_code == 200:
        data = get_request.json()
        if 'Total_score' in data:
            answer_value = data['Total_score']
        else:
            st.write("La réponse ne contient pas de clé 'Total_score'")
    else:
        st.write("La requête GET a échoué avec le code de statut :", get_request.status_code)
    
    answer_value = bool(get_request.json()['Total_score'])
    ### Afficher les résultats ###
    st.write(f'Demande de prêt ID: {selected_credit}')
    st.write(f'Probabilité de défauts de remboursement: {score_client*100:.2f} %')
    # Préparation des données à afficher dans la jauge ==============================================

    # ============== Score du client en pourcentage ==> en utilisant le modèle ======================
    # Sélection des variables du clients
    # Score du client en pourcentage arrondi et nombre entier
    treshold = 0.5
    if score_client < treshold :
        st.write('Demande de prêt acceptée!')
    else:
        #### ajouter une autre condition en fonction de la valeur de la prédiction ####
        if score_client > treshold and score_client <= 0.53:
            st.write('Demande de prêt refusée --> à discuter avec le conseiller')
        else:
            st.write('Demande de prêt refusée!')
    ### ajouter une jauge pour la valeur de prédiction avec la bibliothèque plotly ###
    fig_gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = float(f'{score_client*100:.1f}'),
    mode = "gauge+number+delta",
    title = {'text': "Score(%)"},
    delta = {'reference': treshold*100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge = {'axis': {'range': [0, 100]},
             'bar': {'color': 'black'},
             'steps' : [
                 {'range': [0, 30], 'color': "darkgreen"},
                 {'range': [30, (treshold*100)], 'color': "lightgreen"},
                 {'range': [(treshold*100),53], 'color': "orange"},
                 {'range': [53, 100], 'color':"red"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': treshold*100}}))
    st.plotly_chart(fig_gauge)
    ### ajouter un extenseur pour plus d'explications sur le résultat de la prédiction ###
    with st.expander('Informations complémentaires'):
        st.write(""" Le retour de l'API de prédiction donne un score entre 0 et 100% qui représente la probabilité de refus de prêt.  \n"""
            """ Trois cas de figure sont alors possibles:  \n """
            """ 1) Le score est en dessous de 50% → la demande de prêt est acceptée.  \n """
            """ 2) Le score est entre 50 et 53% → la demande de prêt est refusée 
            mais peut être discutée avec le conseiller pour éventuellement l'accepter 
            (grâce notamment a l'onglet 'analyse des features clients').  \n"""
            """3) Le score est au dessus de 53% → la demande de prêt est refusée. """)










