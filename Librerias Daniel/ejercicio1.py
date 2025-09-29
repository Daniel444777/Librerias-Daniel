import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


fechas = pd.date_range(start="2022-01", periods=24, freq="M")
ventas = np.random.randint(200, 500, size=24)
df = pd.DataFrame({"Fecha": fechas, "Ventas": ventas})
df.set_index("Fecha", inplace=True)


modelo = ARIMA(df["Ventas"], order=(2,1,2))
ajuste = modelo.fit()


pronostico = ajuste.forecast(steps=6)


plt.plot(df.index, df["Ventas"], label="Histórico")
plt.plot(pronostico.index, pronostico, label="Pronóstico", color="red")
plt.legend(); plt.show()
