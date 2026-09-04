import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


queue_data = pd.read_csv("GI_Interactive_Queue.csv")
sheet1_data = pd.read_csv("sheet1.csv")
xtern_data = pd.read_csv("xterndata.csv")
df_data_queue_data = pd.DataFrame(queue_data)
df_data_sheet1= pd.DataFrame(sheet1_data)
df_data_xtern_data = pd.DataFrame(xtern_data)
print(df_data_queue_data)
df_data_xtern_data_filled = df_data_sheet1.fillna(0)
print(df_data_xtern_data_filled)
print(df_data_sheet1.dtypes)
print(df_data_xtern_data)
df_data_queue_data.head()
df_data_queue_data.columns
print(df_data_queue_data.columns)
print(df_data_sheet1.columns)
print(df_data_xtern_data.columns)
df_data_queue_data.drop(['Negotiated In Service Date','Withdrawn Date','POI Name','Generating Facility'], axis=1, inplace=True)
model = RandomForestClassifier()