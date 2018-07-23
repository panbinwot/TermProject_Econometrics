# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 17:32:02 2018

@author: pb061
"""

from FraudDetection import Logreg,FraudForest,FraudTree
import pandas as pd
import numpy as np
import datetime
nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(nowTime+"进行的实验！加油！")


data1 = './temp20.xlsx'
data2 = './temp2withoutyear.xlsx'

folds = 5

#reg = Logreg(data2)
#print(reg.validate(folds))

#reg =FraudTree(data2)
#print(reg.validate(folds))
 
forest = FraudForest(data1)
print(forest.validate(folds))

