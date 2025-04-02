import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from mlxtend.plotting import plot_decision_regions

tempC = []
windspeedKmph = []
precipMM = []
humidity = []
visibility = []
pressure = []
cloudcover = []
HeatIndexC = []
DewPointC = []
WindChillC = []
WindGustKmph = []
FeelsLikeC = []
uvIndex = []

def main():
    for i in range(1, 12):
        getData('2022-%s-01' % i, '2022-%s-01' % (i+1)) #Llamamos a la funcion getData para obtener los datos de la api para cada mes
    df = createDataFrame()
    print(df.head())
    df.to_csv('weather2022.csv', index=False) # guardamos el pandas Dataframe en un csv por si acaso
    #La unica depuracion que tenemos que hacer en el dataframe
    df['precipMM'] = df['precipMM'].astype(float) #Convertimos la columna precipMM a float
    # Discretizamos la clase
    clase = df['precipMM'].round(1)>0.0
    df.loc[clase, 'precipMM'] = 1
    df.loc[~clase, 'precipMM'] = 0
    df = df.astype('int64')
    print('proporcion de precipitaciones')
    print(np.sum(df['precipMM'])/len(df['precipMM']))
    df.to_csv('weather2022disc.csv', index=False) # guardamos el pandas Dataframe en un csv por si acaso
    #Como se trata de un dataset extraido de una api y creado por nosotros ya sabemos que no tenemos que hacerle un tratamiento previo
    #Normalizamos los datos
    df_normalizado = df.loc[:, df.columns != 'precipMM'].apply(normalize, axis=0)
    df_normalizado['precipMM'] = df['precipMM']

    #Hacemos un Hold-out 80-20 para entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(df_normalizado.drop(['precipMM'], axis = 1), df_normalizado['precipMM'], test_size=0.2, random_state=0)
    #Entrenamos el modelo
    model = GaussianNB()
    model.fit(X_train, y_train)
    #Testeamos
    pred = model.predict(X_test)
    print("Tasa de acierto:",metrics.accuracy_score(y_test, pred))
    #Una graficas
    # ConfusionMatrixDisplay.from_estimator(model, X_test, y_test).plot()
    

def getData(data1, data2):
    url = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=66ba03ff2d45466c8aa102127231201&q=Donostia&format=json&date=%s&enddate=%s&tp=3' % (data1, data2)
    response = requests.get(url)
    json = response.json()
    data = json['data']['weather']

    for i in range(len(data)):
        for j in range(0,8):
            tempC.append(data[i]['hourly'][j]['tempC'])
            windspeedKmph.append(data[i]['hourly'][j]['windspeedKmph'])
            precipMM.append(data[i]['hourly'][j]['precipMM'])
            humidity.append(data[i]['hourly'][j]['humidity'])
            visibility.append(data[i]['hourly'][j]['visibility'])
            pressure.append(data[i]['hourly'][j]['pressure'])
            cloudcover.append(data[i]['hourly'][j]['cloudcover'])
            HeatIndexC.append(data[i]['hourly'][j]['HeatIndexC'])
            DewPointC.append(data[i]['hourly'][j]['DewPointC'])
            WindChillC.append(data[i]['hourly'][j]['WindChillC'])
            WindGustKmph.append(data[i]['hourly'][j]['WindGustKmph'])
            FeelsLikeC.append(data[i]['hourly'][j]['FeelsLikeC'])
            uvIndex.append(data[i]['hourly'][j]['uvIndex'])

def createDataFrame():
    df = pd.DataFrame({'tempC': tempC,
                       'windspeedKmph': windspeedKmph,
                       'precipMM': precipMM,
                       'humidity': humidity,
                       'visibility': visibility,
                       'pressure': pressure,
                       'cloudcover': cloudcover,
                       'HeatIndexC': HeatIndexC,
                       'DewPointC': DewPointC,
                       'WindChillC': WindChillC,
                       'WindGustKmph': WindGustKmph,
                       'FeelsLikeC': FeelsLikeC,
                       'uvIndex': uvIndex})
    return df

def normalize(x):
    return((x-min(x)) / (max(x) - min(x)))

if __name__ == '__main__':
    np.random.seed(2)
    main()

