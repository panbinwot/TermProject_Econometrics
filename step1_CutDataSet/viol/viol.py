# -*- coding: utf-8 -*-
"""
Created on Sun May 27 12:25:16 2018

@author: Panbin
Project: Predicting Accounting Misstatement
Term Paper for Econometrics II
"""
path= "C:/Users/lenovo/Desktop/termpapering/step1_CutDataSet"

import pandas as pd
import numpy as np
import math

f = pd.read_excel(path+"/"+"STK_Violation_Main.xlsx")
df = f[['Symbol','DisposalDate','IsViolated','ViolationTypeID','Year']]

for m in range(0,len(df)):
    if math.isnan(df.iat[m,4]):
        df.iat[m,4] = df.iat[m,1].split("-")[0]

for i in range(2,9):
    x = str(i)
    ddf = f[['Symbol','DisposalDate','IsViolated','ViolationTypeID','Year'+x]]
    ddf = ddf.dropna(axis=0,how='any')
    ddf.rename(columns={'Year'+x:'Year'}, inplace=True)
    df = pd.concat([df,ddf])
df['lemon'] = 1    
f0 = df[['Symbol','Year','lemon']]
f0 = f0.drop_duplicates()

f0['Year'] = pd.to_numeric(f0['Year'],errors='coerce')
f0 = f0.loc[f0['Year']>=2000]
f0.to_excel(path+"/"+"violition_list.xlsx",index = False)

#type(f0.iat[1,1])