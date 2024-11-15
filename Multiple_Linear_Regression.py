import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Büyüklük (m2)": [100,150,200,120,180,90],
    "Oda Sayısı": [2, 3, 4, 2, 4, 2],
    "Bina Yaşı": [10, 15, 20, 12, 18, 8],
    "Fiyat (bin TL)": [300, 450, 600, 250, 520, 280]
}

df = pd.DataFrame(data)

X = df[["Büyüklük (m2)", "Oda Sayısı", "Bina Yaşı"]]
y = df["Fiyat (bin TL)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize = (6,4))
plt.scatter(range(len(y_test)), y_test, color = "blue", label = "Gerçek Fiyatlar", s = 100)
plt.scatter(range(len(y_pred)), y_pred, color = "red", label = "Tahmin Edilen Fiyatlar", s = 100)
plt.xlabel("Örnekler")
plt.ylabel("Fiyat (bin TL)")
plt.title("Gerçek Ve Tahmin Edilen Fiyatlar")
plt.legend()
plt.grid(True)
plt.show()
