#keşifsel veri analizi (EDA) 
import numpy as np #linear algebra(matematik işlemleri)
import pandas as pd #data processing(cvs dosya okuması)
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px

import missingno as msno #missing value 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,RepeatedStratifiedKFold,train_test_split
from sklearn.metrics import precision_score,confusion_matrix

from sklearn import tree

df= pd.read_csv("water_potability.csv")

describe = df.describe()
info = df.info()
print(info)
 


#preprocessing:missing value problem,train-test split,normalization,encoding

#modelleme: decision tree, random forest

#evaluation: decision tree visualization

#hyperparameter tuning:  random forest
