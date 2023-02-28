#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:24:12 2023

@author: JulianDPR
"""
###Install packages

# Linux: pip install numpy pandas princes matplotlib distfit scipy seaborn sklearn
# Windows: pip install princes matplotlib scipy seaborn sklearn

###Import packages

import numpy as np
import pandas as pd
import prince as p 
import matplotlib.pyplot as plt
import distfit.distfit as df
import scipy.stats as ss
import seaborn as sns
from sklearn.linear_model import LinearRegression
from mpl_toolkits import mplot3d

##########################################

### Lectura bases de datos

grasa = pd.read_csv("in/grasa.csv",sep=",")
missing = pd.read_csv("in/Rmissing.csv", sep=",",index_col=0)
wine = pd.read_csv("in/wine.csv", sep=",")
del wine["Judge"]

##########################################

### missing

fig, ax = plt.subplots()
missing1 = missing.loc[missing.loc[:,"mydata"]!=999,:]
missing1 = missing1.loc[pd.notna(missing["mydata"]),:]
missing1["mydata"].plot(ax=ax,kind="hist", color="coral")
missing1["mydata"].plot(ax=ax,kind="kde",secondary_y=True, color="deepskyblue")
df().fit_transform(missing1.iloc[:,0])
ax.set_title("Histograma de datos pre-simulación")
ax.set_xlabel("Datos", fontweight="bold")
ax.set_ylabel("Frecuencias", fontweight="bold")

fig.savefig("out/missing/datosimuladospre.png", dpi=500)
fig.show()

#Observamos que es muy probable que es normal, tanto por los residuos.
#Entonces procedemos a simular esos datos

long1 = len(missing.loc[missing["mydata"]==999,:])
long2 = len( missing.loc[missing["mydata"].isna(),:])

np.random.seed(137)

#Se crean datos aleatorios para completar la muestra
mu = missing1.mean().to_numpy()[0]
sd = missing1.var().to_numpy()[0]**(1/2)

missing.loc[missing["mydata"]==999,:]=np.random.normal(mu,sd,long1).reshape(long1,1)
missing.loc[missing["mydata"].isna(),:]=np.random.normal(mu,sd,long2).reshape(long2,1)

np.random.seed(None)

fig, ax = plt.subplots()
missing["mydata"].plot(ax=ax,kind="hist",color='teal')
missing["mydata"].plot(ax=ax,kind="kde",secondary_y=True, color='tomato')
df().fit_transform(missing.iloc[:,0])
ax.set_title("Histograma de datos post-simulación")
ax.set_xlabel("Datos", fontweight="bold")
ax.set_ylabel("Frecuencias", fontweight="bold")

fig.savefig("out/missing/datossimulados.png", dpi=500)
fig.show()

########################################

### Vinos

col = wine.columns
col = list(col)
wine1 = wine
wine = wine.to_numpy().transpose()
wine2 = np.zeros((len(col),len(col)))
wine2 = pd.DataFrame(np.matrix(wine2), index=col, columns=col)

print("\n","Friedman: ","\n",np.round(ss.friedmanchisquare(*wine),3))

#Se aplica correcion de Bonferroni
anew = 0.05/10

# definimos una función
def testwilcoxon(x,y):
  stat, p = ss.wilcoxon(x, y, alternative='two-sided')
  print(f'Statistics={stat}, p={p}')
  # interpretación
  alpha=0.05/10
  if p > alpha:
    print('No rechazamos H0: No hay diferencias significativas')
  else:
    print('Rechazamos H0: Hay diferencias significativas')
  return(p)  
    
for i in range(len(col)):
    
    wine2.loc[col[i], col[i]] = 1
    
    for j in range(i, len(col)-1):
        
        print(col[i], col[j+1])
        
        a = testwilcoxon(wine1[col[i]], wine1[col[j+1]])
        
        wine2.loc[col[i], col[j+1]] = a
        wine2.loc[col[j+1], col[i]] = a

fig = plt.figure()

ax = fig.add_subplot(111)

ax = sns.heatmap(wine2, annot=True)

ax.set_title("Matriz de valores p")

fig.show()
        
##################################

### Grasa

cgrasa = grasa - grasa.mean().transpose()
corrgrasa = cgrasa.corr()
grasa.plot(kind="box")
plt.title("Cajas y bigotes variables")
plt.savefig("out/grasa/cajas.png", dpi=600)
plt.show()

sns.heatmap(corrgrasa, annot=True)
plt.title("Correlación variables")
plt.savefig("out/grasa/calor.png", dpi=600)
plt.show()

pd.plotting.scatter_matrix(grasa, alpha=0.5)
plt.savefig("out/grasa/puntos.png", dpi=1500)
plt.show()


##Regresion grasa

model = LinearRegression()

x = grasa.loc[:,["siri", "bmi"]].to_numpy()
y = grasa.loc[:, "abdomen"].to_numpy()

model = model.fit(x,y)

r_sq = model.score(x,y)

print(f"El coeficiente de determinación: {r_sq}", "\n",
      f"intercepción: {model.intercept_}", "\n",
      f"coeficiente: {model.coef_}")

y_pred = model.predict(x)

print(f"Predicción: {y_pred}")

x_surf , y_surf = np.meshgrid(np.linspace(grasa.siri.min(),grasa.siri.max(),
                                          100),
                              np.linspace(grasa.bmi.min(),
                                           grasa.bmi.max(),100))

onlyX = pd.DataFrame({"siri":x_surf.ravel(), "bmi":y_surf.ravel()})

y_pred = model.predict(onlyX)

fig = plt.figure()

my_cmap = plt.get_cmap("hsv")

ax = fig.add_subplot(111, projection="3d")

sctt = ax.scatter3D(grasa["siri"], grasa["bmi"], grasa["abdomen"], c=grasa["siri"]+
           grasa["bmi"]+grasa["abdomen"], cmap = my_cmap, alpha=0.5)

trisurf = ax.plot_trisurf(x_surf.ravel(), y_surf.ravel(),
                y_pred, cmap = my_cmap,
                alpha=0.3)

ax.set_title("Modelo lineal bmi, siri, abdomen")
ax.set_xlabel("siri", fontweight = "bold")
ax.set_ylabel("bmi", fontweight = "bold")
ax.set_zlabel("abdomen", fontweight = "bold")

fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)

fig.savefig("out/grasa/model.png", dpi=600)

fig.show()