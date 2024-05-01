import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Primer podatkov - predpostavljamo, da imamo zgodovinske podatke o prodaji
# X predstavlja neodvisne spremenljivke (npr. cene, oglaševalski proračun, itd.)
# y predstavlja odvisno spremenljivko (npr. število prodanih izdelkov)
data = {
    'Cena': [100, 200, 300, 400, 500],
    'Oglaševalski_proračun': [50, 100, 150, 200, 250],
    'Prodaja': [150, 300, 450, 400, 600]
}

df = pd.DataFrame(data)

# ločitev podatkov na vektorje funkcij (X) in ciljno spremenljivko (y)
X = df[['Cena', 'Oglaševalski_proračun']]
y = df['Prodaja']

# razdelitev podatkov na učni in testni nabor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ustvarjanje modela linearne regresije
model = LinearRegression()

# Učenje modela na učnih podatkih
model.fit(X_train, y_train)

# Napovedovanje na testnih podatkih
y_pred = model.predict(X_test)

# Izpis rezultatov
print("Napovedane vrednosti:", y_pred)
print("Prave vrednosti:", y_test.values)

# Shranite graf kot sliko
plt.scatter(y_test, y_pred)
plt.xlabel("Prave vrednosti")
plt.ylabel("Napovedane vrednosti")
plt.title("Napovedovanje povpraševanja - Model linearnega regresije")
plt.savefig('napovedovanje_graf.png')  # Shranite graf kot sliko

