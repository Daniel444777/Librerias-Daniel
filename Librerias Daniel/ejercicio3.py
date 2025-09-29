import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")


tweets = [
    "Me encanta este producto, es increíble",
    "No me gustó el servicio, fue terrible",
    "Está bien, pero podría mejorar",
    "Excelente atención, volveré a comprar",
    "Pésima calidad, no lo recomiendo"
]

df = pd.DataFrame({"Tweet": tweets})

sia = SentimentIntensityAnalyzer()
df["Sentimiento"] = df["Tweet"].apply(lambda x: sia.polarity_scores(x)["compound"])


df["Etiqueta"] = df["Sentimiento"].apply(lambda x: "Positivo" if x>0 else ("Negativo" if x<0 else "Neutral"))
print(df)
