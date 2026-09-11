from Read_Data.open_csv import read_data
from Write_Data.model import model
from Write_Data.data_manipulation import manipulate

data = read_data()

df_data_queue_data , df_data_sheet1, df_data_int_queue = data.read_file()

man_data = manipulate(df_data_queue_data, df_data_int_queue)
man_data.filter_data()

large_load_data = man_data.extract_dates()

Model = model(df_data_queue_data,large_load_data)

man_data.log_mw()
preprocessor = Model.model_feature_selection()

x, y = Model.model_predict(preprocessor)
