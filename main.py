from Read_Data.open_csv import read_data
from Write_Data.model import model
from Write_Data.data_manipulation import manipulate


data = read_data()

df_data_queue_data, df_data_int_queue = data.read_file()

man_data = manipulate(
    df_data_queue_data,
    df_data_int_queue
)

man_data.filter_data()

large_load_data = man_data.extract_dates()

Model = model(
    large_load_data
)

man_data.log_mw()

preprocessor = Model.model_feature_selection()

pipeline, expected_mw = Model.model_predict(
    preprocessor
)


# =========================================================
# WRITE EXPECTED MW DATAFRAME TO CSV
# =========================================================

expected_mw.to_csv(
    "expected_mw.csv",
    index=False
)


# =========================================================
# PRINT DATAFRAME
# =========================================================

print(expected_mw)
