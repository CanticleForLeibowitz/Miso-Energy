import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


queue_data = pd.read_csv('csv data\GI_Interactive_Queue.csv')
sheet1_data = pd.read_csv("csv data\sheet1.csv")
df_data_queue_data = pd.DataFrame(queue_data)
df_data_sheet1= pd.DataFrame(sheet1_data)
print(df_data_queue_data)
df_data_xtern_data_filled = df_data_sheet1.fillna(0)
print(df_data_xtern_data_filled)
print(df_data_sheet1.dtypes)
df_data_queue_data.head()
df_data_queue_data.columns
print(df_data_queue_data.columns)
print(df_data_sheet1.columns)
df_data_queue_data.drop(['Negotiated In Service Date','Withdrawn Date','POI Name','Generating Facility'], axis=1, inplace=True)
X = df_data_sheet1.drop(columns=['INCREMENTAL_LOAD_MW','STATE']).apply(pd.to_numeric, errors='coerce')
y =  pd.to_numeric(df_data_sheet1['INCREMENTAL_LOAD_MW'], errors="coerce")
y = y.fillna(0)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, predictions))
