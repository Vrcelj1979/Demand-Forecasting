import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from PIL import Image
import io

# Example data - suppose we have historical sales data
# X represents the independent variables (eg prices, advertising budget, etc.)
# y represents the dependent variable (e.g. number of products sold)
data = {
    'Cena': [100, 200, 300, 400, 500],
    'Oglaševalski_proračun': [50, 100, 150, 200, 250],
    'Prodaja': [150, 300, 450, 400, 600]
}

df = pd.DataFrame(data)

# data separation into function vectors (X) and target variable (y)
X = df[['Cena', 'Oglaševalski_proračun']]
y = df['Prodaja']

# splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Learning the model on the training data
model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

# Create a graph
plt.scatter(y_test, y_pred)
plt.xlabel("Prave vrednosti")
plt.ylabel("Napovedane vrednosti")
plt.title("Napovedovanje povpraševanja - Model linearnega regresije")

# Convert the graph to an image and display it in the terminal
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = Image.open(buf)
image.show()
