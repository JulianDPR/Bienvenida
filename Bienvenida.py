#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:24:12 2023

@author: JulianDPR
"""
###Install packages

# Linux: pip install numpy pandas princes matplotlib distfit scipy
# Windows: pip install princes matplotlib scipy

###Import packages

import numpy as np
import pandas as pd
import prince as p 
import matplotlib.pyplot as plt
import distfit.distfit as df
import scipy.stats as ss


grasa = pd.read_csv("in/grasa.csv",sep=",")
missing = pd.read_csv("in/Rmissing.csv", sep=",",index_col=0)
wine = pd.read_csv("in/wine.csv", sep=",")
del wine["Judge"]

#Inicio por missing
fig, ax = plt.subplots()
missing1=missing.loc[missing.loc[:,"mydata"]!=999,:]
missing1=missing1.loc[pd.notna(missing["mydata"]),:]
missing1["mydata"].plot(ax=ax,kind="hist")
missing1["mydata"].plot(ax=ax,kind="kde",secondary_y=True)
df().fit_transform(missing1.iloc[:,0])

fig.savefig("out/datosimuladospre.png", dpi=500)
fig.show()

#Observamos que es muy probable que es normal, tanto por los residuos.
#Entonces procedemos a simular esos datos

long1 = len(missing.loc[missing["mydata"]==999,:])
long2 = len( missing.loc[missing["mydata"].isna(),:])

np.random.seed(1)

#Se crean datos aleatorios para completar la muestra

missing.loc[missing["mydata"]==999,:]=np.random.normal(32.971,11.326,long1).reshape(long1,1)
missing.loc[missing["mydata"].isna(),:]=np.random.normal(32.971,11.326,long2).reshape(long2,1)

np.random.seed(None)

fig, ax = plt.subplots()
missing["mydata"].plot(ax=ax,kind="hist")
missing["mydata"].plot(ax=ax,kind="kde",secondary_y=True)
df().fit_transform(missing.iloc[:,0])

fig.savefig("out/datossimulados.png", dpi=500)
fig.show()

########################################

### Vinos
col = wine.columns
col = list(col)
wine1 = wine
wine = wine.to_numpy().transpose()

print(np.round(ss.friedmanchisquare(*wine),3))

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
    
for i in range(len(col)):
    
    for j in range(i, len(col)-1):
        
        print(col[i], col[j+1])
        
        testwilcoxon(wine1[col[i]], wine1[col[j+1]])
        
#################


        

    
