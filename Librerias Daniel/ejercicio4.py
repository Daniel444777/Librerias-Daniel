import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


np.random.seed(42)
data = {
    "Temp": np.random.randint(50,100,100),
    "Vibracion": np.random.randint(10,50,100),
    "Presion": np.random.randint(1,10,100),
    "Falla": np.random.choice([0,1], size=100) # 0=OK, 1=Falla
}
df = pd.DataFrame(data)


X = df.drop("Falla",axis=1)
y = df["Falla"]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
modelo = RandomForestClassifier()
modelo.fit(X_train,y_train)
pred = modelo.predict(X_test)

print("Precisión:", accuracy_score(y_test,pred))
