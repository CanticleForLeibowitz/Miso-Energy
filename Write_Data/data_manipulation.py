import numpy as np
import pandas as pd

class manipulate:
    def __init__(self, data_set1, data_set2):
        self.data_set1 = data_set1
        self.data_set2 = data_set2

    def drop_columns(self):
        self.data_set1.drop(['Negotiated In Service Date','Withdrawn Date','POI Name','Generating Facility'], axis=1, inplace=True)
        self.data_set2.drop(columns=['INCREMENTAL_LOAD_MW','STATE']).apply(pd.to_numeric, errors='coerce')

    def filter_data(self):
        active = self.data_set1[self.data_set1['Request Status'] == 'Active']
        done = self.data_set1[self.data_set1['Request Status'] == 'Done']
        withdrawn = self.data_set1[self.data_set1['Request Status'] == 'Withdrawn'] 
        legacy = self.data_set1[self.data_set1['Request Status'] == 'LEGACY: Done']
        print("Active Records:")
        print(len(active))
        print("Done Records:")
        print(len(done))
        print("Withdrawn Records:")
        print(len(withdrawn))
        print("Legacy Done")
        print(len(legacy))