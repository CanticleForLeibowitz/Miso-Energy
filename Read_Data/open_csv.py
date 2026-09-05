import pandas as pd

class read_data:
    def __init__(self):
        self.queue_file = 'csv_data\GI_Interactive_Queue.csv'
        self.sheet_file =  'csv_data\sheet1.csv'

    def read_file(self):
        queue_data = pd.read_csv(f'{self.queue_file}')
        sheet1_data = pd.read_csv(f'{self.sheet_file}')
        return queue_data , sheet1_data
