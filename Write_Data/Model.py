from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


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

        # Numeric features
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical features
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ])

        # Combine preprocessing
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

        # Create Logistic Regression pipeline
        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=2000
            ))
        ])

        # --------------------------------------------------
        # 1. Create realized variable
        # --------------------------------------------------

        self.data_set2 = self.data_set2.copy()

        self.data_set2["realized"] = (
            self.data_set2["q_status"].map({
                "operational": 1,
                "withdrawn": 0
            })
        )

        # --------------------------------------------------
        # 2. Training data
        # --------------------------------------------------
        # Only use projects where we know the outcome.
        #
        # operational = 1
        # withdrawn   = 0
        # everything else = NaN
        #
        # We cannot train Logistic Regression on NaN outcomes.

        train = self.data_set2[
            (self.data_set2["q_year"] <= 2018) &
            (self.data_set2["realized"].notna())
        ].copy()

        # --------------------------------------------------
        # 3. Test data
        # --------------------------------------------------

        test = self.data_set2[
            self.data_set2["q_year"] > 2018
        ].copy()

        # --------------------------------------------------
        # 4. Select features and target
        # --------------------------------------------------

        X_train = train[features]
        y_train = train["realized"]

        X_test = test[features]

        # --------------------------------------------------
        # 5. Train model
        # --------------------------------------------------

        model_pipeline.fit(X_train, y_train)

        # --------------------------------------------------
        # 6. Predict probability of realization
        # --------------------------------------------------

        probabilities = model_pipeline.predict_proba(X_test)[:, 1]
        print(probabilities)
        # --------------------------------------------------
        # 7. Add predictions to test dataset
        # --------------------------------------------------

        test["realization_probability"] = probabilities

        # --------------------------------------------------
        # 8. Calculate expected MW
        # --------------------------------------------------

        test["expected_mw"] = (
            test["total_mw"] *
            test["realization_probability"]
        )

        # --------------------------------------------------
        # 9. Return results
        # --------------------------------------------------

        return model_pipeline, test


    def forrecast(self, MW, probability):

        Forecasted_Load = MW * probability

        return Forecasted_Load