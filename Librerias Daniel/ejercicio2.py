import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


data = {
    "Edad": [25,40,35,50,23,30,45,60,32,41],
    "Meses_Cliente": [12,48,36,60,5,15,50,70,10,55],
    "Uso_Servicio": [20,200,150,300,10,40,250,400,15,280],
    "Churn": [0,0,0,1,1,0,1,1,1,0]
}
df = pd.DataFrame(data)


X = df.drop("Churn", axis=1)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
modelo = LogisticRegression()
modelo.fit(X_train,y_train)
pred = modelo.predict(X_test)

print(classification_report(y_test,pred))
