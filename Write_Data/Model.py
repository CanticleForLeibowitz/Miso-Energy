from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

class model:
    def __init__(self, data_set1, data_set2):
        self.data_set1 = data_set1
        self.data_set2 = data_set2


    def model_feature_selection(self):
        
        numeric_features = [
            "log_mw",
            "planned_years"
        ]
        categorical_features = [
            "type_clean",
            "region",
            "service",
            "IA_phase_clean"
        ]
        numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

        categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore"
                ))
             ])
        
        preprocessor = ColumnTransformer([
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features)
             ])
        return preprocessor


    def model_predict(self, preprocessor):
        features = [
                    "log_mw",
                    "planned_years",
                    "type_clean",
                    "region",
                    "service",
                    "IA_phase_clean"
                ]
        model = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    max_iter=2000
                ))
            ])
        self.data_set2["realized"] = self.data_set2["q_status"].map({
                    "operational": 1,
                    "withdrawn": 0
                })
        train = self.data_set2[self.data_set2["q_year"] <= 2018]
        test = self.data_set2[self.data_set2["q_year"] > 2018]        
        X_train = train[features]
        y_train = train["realized"]

        X_test = test[features]
        y_test = test["realized"]

        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        test = test.copy()

        test["realization_probability"] = probabilities
        test["expected_mw"] = (
        test["total_mw"] *
        test["realization_probability"]
    )
        #realized = 1 if q_status = 'operational'
        #realized = 0 if q_status = 'withdrawn'
       # realized = NA
        #total_mw = mw1 ​+mw2 ​+ mw3
        #planned_lead_time = prop_date − q_date
        #time_to_operation = on_date − q_date
        #time_to_withdrawal = wd_date−q_date
        #location × project size
        #project age = current date − q_date



   

    def forrecast(self, MW ,probability):
        Forecasted_Load = MW * probability
        return Forecasted_Load
        #Forecasted_Load = Requested MW * Probability of Realization
