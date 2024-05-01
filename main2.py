import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

# Print the results
print("Napovedane vrednosti:", y_pred)
print("Prave vrednosti:", y_test.values)

# Save the graph as an image
plt.scatter(y_test, y_pred)
plt.xlabel("Prave vrednosti")
plt.ylabel("Napovedane vrednosti")
plt.title("Napovedovanje povpraševanja - Model linearnega regresije")
plt.savefig('napovedovanje_graf.png')  # Shranite graf kot sliko

