from Read_Data.open_csv import read_data
from Write_Data.model import model
from Write_Data.data_manipulation import manipulate

data = read_data()

df_data_queue_data , df_data_sheet1 = data.read_file()


man_data = manipulate(df_data_queue_data, df_data_sheet1)
man_data.filter_data()

