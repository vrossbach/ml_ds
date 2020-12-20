# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:20:41 2020

@author: vivienrossbach
"""

import pandas as pd

base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values

classe = base.iloc[:, 14].values

# Transformar atributo nominal em discreto

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsores = LabelEncoder()
# labels = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

# Variáveis do tipo dummy

previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
ct = ColumnTransformer([("race", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder = 'passthrough')
previsores = ct.fit_transform(previsores)

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)


