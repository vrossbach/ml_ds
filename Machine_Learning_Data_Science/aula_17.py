# -*- coding: utf-8 -*-
"""
Base de dados de crédito
"""
import pandas as pd
base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age'] < 0]
# Apagar a coluna inteira:
# base.drop('age', 1, inplace=True) # 1 apaga tudo, inplace para não retornar nada

# As alterações são feitas no DataFrame e o csv não é alterado.

# Apagar somente os registros com problema:
# base.drop(base[base.age < 0].index, inplace)  # index apaga os índices dos atributos

# Preencher os valores manualmente

# Preencher os valores com a média:
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

# Localizar valores faltantes (True):
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]


# Dividir por tipos de variáveis
previsores = base.iloc[:, 1:4].values # todas as linhas do atributo 1 ao 3 (a primeira coluna é o 0)

classe = base.iloc[:, 4].values

# Remover valores nulos
import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Aqui instanciamos a classe e usamos seus atributos depois do ponto (.)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

# Escalonamento (importante no KNN - distância euclidiana)
# Aumenta a velocidade de execução
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
# A classe não precisa ser padronizada porque só tem 2 valores, 0 e 1
