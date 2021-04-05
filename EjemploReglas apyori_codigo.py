# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:17:37 2021

@author: Casa
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

market = pd.read_csv('./Market_Basket_Optimisation.csv',header=None)

market.head()

num_registros = len(market)

registros = []
#for i in range(0, num_registros):
#    registros.append([str(market.values[i,j]) for j in range(0,20)])
    
for sublist in market.values.tolist():
  clean_sublist = [item for item in sublist if item is not np.nan]
  registros.append(clean_sublist)


reglas_asociciacion = apriori(registros,min_support=0.005,min_confidence=0.20,min_lift=3,min_lenght=3)
resultado_asociciacion = list(reglas_asociciacion)

resultados = []
for item in resultado_asociciacion:
    pair = item[0]
    items = [x for x in pair]
    
    value0 = str(items[0])
    value1 = str(items[1])
    value2 = str(item[1])[:5]
    value3 = str(item[2][0][2])[:5]
    value4 = str(item[2][0][3])[:5]
    
    rows = (value0,value1,value2,value3,value4)
    
    resultados.append(rows)
    
    Label = ['Title1','Title2','Support','Confidence','Lift']
    
    sugerencias = pd.DataFrame.from_records(resultados,columns=Label)
