from flask import Flask, render_template, request, send_from_directory
import pandas as pd 
import numpy as np 
import requests
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hasil', methods=['GET', 'POST'])
def hasil():
    body = request.form 

    if body['digimon'].lower() not in df['lower'].values:
        return render_template('error.html')
    else:
        fav = df[df['lower'] == body['digimon'].lower()]

        score = list(enumerate(cosScore[fav.index[0]]))
        score = sorted(score, key=lambda i: i[1], reverse=True)

        recommendation = []
        for i in score[:7]:
            if i[0] != fav.index[0]:
                recommendation.append(i[0])

        return render_template('hasil.html', df=df, recommendation=recommendation, fav=fav.index[0])

if __name__ == '__main__':

    with open('digimon.json', 'r') as x:
        data = json.load(x)
    
    df = pd.DataFrame(data)

    df['x'] = df.apply(
        lambda i: f"{i['stage']},{str(i['type'])},{str(i['attribute'])}",
        axis = 1
    )
    df = df[['digimon', 'image', 'stage', 'type', 'attribute', 'x']]

    df['lower'] = df.apply(lambda i: i['digimon'].lower(), axis=1)

    from sklearn.feature_extraction.text import CountVectorizer
    model = CountVectorizer(
        tokenizer = lambda i: i.split(',')
    )
    gMatrix = model.fit_transform(df['x'])

    from sklearn.metrics.pairwise import cosine_similarity
    cosScore = cosine_similarity(gMatrix)


    app.run(debug = True)