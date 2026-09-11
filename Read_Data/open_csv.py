import pandas as pd

class read_data:
    def __init__(self):
        self.queue_file = 'csv_data\GI_Interactive_Queue.csv'
        self.sheet_file =  'csv_data\sheet1.csv'
        self.int_queue = 'csv_data\Interconnection_Queue_Dataset.csv'

    def read_file(self):
        queue_data = pd.read_csv(f'{self.queue_file}')
        sheet1_data = pd.read_csv(f'{self.sheet_file}')
        int_queue = pd.read_csv(f'{self.int_queue}',header=1)
        int_queue.columns = int_queue.columns.str.strip()
        print(int_queue)
        return queue_data , sheet1_data, int_queue
