from flask import Flask, send_file
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route('/')
def show_graph():
    # Generiranje grafa
    data = {
        'Cena': [100, 200, 300, 400, 500],
        'Oglaševalski_proračun': [50, 100, 150, 200, 250],
        'Prodaja': [150, 300, 450, 400, 600]
    }
    df = pd.DataFrame(data)
    X = df[['Cena', 'Oglaševalski_proračun']]
    y = df['Prodaja']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Prave vrednosti")
    plt.ylabel("Napovedane vrednosti")
    plt.title("Napovedovanje povpraševanja - Model linearnega regresije")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
