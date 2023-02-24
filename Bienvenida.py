#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:24:12 2023

@author: JulianDPR
"""
###Install packages

# Linux: pip install numpy pandas princes matplotlib distfit
# Windows: pip install princes matplotlib

###Import packages

import numpy as np
import pandas as pd
import prince as p 
import matplotlib.pyplot as plt
import distfit.distfit as df


grasa = pd.read_csv("in/grasa.csv",sep=",")
missing = pd.read_csv("in/Rmissing.csv", sep=",",index_col=0)
wine = pd.read_csv("in/wine.csv", sep=",")

#Inicio por missing
fig, ax = plt.subplots()
missing1=missing.loc[missing.loc[:,"mydata"]!=999,:]
missing1=missing1.loc[pd.notna(missing["mydata"]),:]
missing1["mydata"].plot(ax=ax,kind="hist")
missing1["mydata"].plot(ax=ax,kind="kde",secondary_y=True)
df().fit_transform(missing1.iloc[:,0])
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
fig.show()