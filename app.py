import os
import pandas as pd 
import numpy as np 
import flask
import pickle as pk
#https://medium.com/techcrush/how-to-deploy-your-ml-model-in-jupyter-notebook-to-your-flask-app-d1c4933b29b5
from flask import Flask, render_template, request

app=Flask(__name__)

def init():
    global data_encoder, neuralNetModel

    print("initializing... ") 
    data_encoder = pk.load(open("/Users/vbono/Documents/computer-science/ai/final-proj/label_encoder_map.pkl", "rb"))
    neuralNetModel = pk.load(open("/Users/vbono/Documents/computer-science/ai/final-proj/nn.pkl", "rb")) 

    print("initialized")


@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods = ['POST'])
def result():

    # incoming request
    to_predict_list = request.form.to_dict()
    to_predict_list=list(to_predict_list.values())
    print(to_predict_list)

    # build the structure to run this data through the model
    to_predict = np.array(to_predict_list).reshape(1,7)
    print(to_predict)

    df = pd.DataFrame(to_predict, columns = [ 'Pclass','Sex','Age','SibSp', 'Parch',   'Fare', 'Embarked']) 
    #set to lowercase
    df= df.applymap(lambda s:s.lower() if type(s) == str else s)

    # recall 
    #note from the original df - only sex,   embarked are strings
    #    [[3,'male', 34.5, 0,0,,7.8292  ,'Q'],

    df['explicit'] = df['explicit'].astype(bool)
    df['energy'] = df['energy'].astype(float)
    df['key'] = df['key'].astype(int) 
    df['loudness'] = df['loudness'].astype(float) 
    df['speechiness'] = df['speechiness'].astype(float)
    df['acousticness'] = df['acousticness'].astype(float)
    df['instrumentalness'] = df['instrumentalness'].astype(float)
    df['liveness'] = df['liveness'].astype(float)
    df['valence'] = df['valence'].astype(float)
    df['tempo'] = df['tempo'].astype(float) 


    #try the encode 
    df.replace(data_encoder, inplace=True)
    
    print ('row after encode: ') # default rows is 5
    print (df.head()) # default rows is 5
    print(df)

    y_pred = neuralNetModel.predict_proba(df)
    print("X=%s, Predicted=%s" % (df.to_numpy, y_pred[0]))
 
    return render_template('predict.html', prediction=y_pred[0])

 
   

if __name__ == '__main__':
    init()
    app.run(debug=True, port=9090)
