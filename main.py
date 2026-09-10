import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from Read_Data.open_csv import read_data
from Write_Data.model import model
from Write_Data.data_manipulation import manipulate

data = read_data()

df_data_queue_data , df_data_sheet1 = data.read_file()
print(df_data_queue_data)
print(df_data_sheet1.dtypes)

df_data_queue_data.drop(['Negotiated In Service Date','Withdrawn Date','POI Name','Generating Facility'], axis=1, inplace=True)
X = df_data_sheet1.drop(columns=['INCREMENTAL_LOAD_MW','STATE']).apply(pd.to_numeric, errors='coerce')
y =  pd.to_numeric(df_data_sheet1['INCREMENTAL_LOAD_MW'], errors="coerce")
y = y.fillna(0)
Model = model(y,X)
man_data = manipulate(df_data_queue_data, df_data_sheet1)
man_data.filter_data()

