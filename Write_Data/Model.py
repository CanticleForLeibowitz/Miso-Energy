from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

import matplotlib.pyplot as plt
import pandas as pd


class model:

    def __init__(self, data_set2):
        self.data_set2 = data_set2

    def model_feature_selection(self):

        # =========================================================
        # NUMERICAL FEATURES
        # =========================================================

        numeric_features = [
            "log_mw",
            "planned_years"
        ]

        # =========================================================
        # CATEGORICAL FEATURES
        # =========================================================

        categorical_features = [
            "type_clean",
            "region",
            "service",
            "IA_phase_clean"
        ]

        # =========================================================
        # NUMERICAL PIPELINE
        # =========================================================

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        # =========================================================
        # CATEGORICAL PIPELINE
        # =========================================================

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        # =========================================================
        # COMBINE PREPROCESSORS
        # =========================================================

        preprocessor = ColumnTransformer([
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ])

        return preprocessor

    def model_predict(self, preprocessor):

        # =========================================================
        # FEATURES
        # =========================================================

        features = [
            "log_mw",
            "planned_years",
            "type_clean",
            "region",
            "service",
            "IA_phase_clean"
        ]

        # =========================================================
        # MODEL PIPELINE
        # =========================================================

        model_pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000
                )
            )
        ])

        # =========================================================
        # COPY DATA
        # =========================================================

        data = self.data_set2.copy()

        # =========================================================
        # TARGET VARIABLE
        # =========================================================

        # Operational = 1
        # Withdrawn = 0
        # Everything else = NaN / unknown

        data["realized"] = data["q_status"].map({
            "operational": 1,
            "withdrawn": 0
        })

        # =========================================================
        # TRAINING DATA
        # =========================================================

        train = data[
            (data["q_year"] <= 2016) &
            (data["realized"].notna())
        ].copy()

        # =========================================================
        # CALIBRATION DATA
        # =========================================================

        calibration = data[
            (data["q_year"].between(2017, 2018)) &
            (data["realized"].notna())
        ].copy()

        # =========================================================
        # TEST DATA
        # =========================================================

        test = data[
            data["q_year"] > 2018
        ].copy()

        # =========================================================
        # TRAIN MODEL
        # =========================================================

        model_pipeline.fit(
            train[features],
            train["realized"]
        )

        # =========================================================
        # CALIBRATION
        # =========================================================

        calibration_prob = (
            model_pipeline
            .predict_proba(
                calibration[features]
            )[:, 1]
        )

        iso = IsotonicRegression(
            y_min=0,
            y_max=1,
            out_of_bounds="clip"
        )

        iso.fit(
            calibration_prob,
            calibration["realized"]
        )

        # =========================================================
        # TEST PREDICTIONS
        # =========================================================

        test["raw_probability"] = (
            model_pipeline
            .predict_proba(
                test[features]
            )[:, 1]
        )

        test["calibrated_probability"] = (
            iso.predict(
                test["raw_probability"]
            )
        )

        # =========================================================
        # EXPECTED MW
        # =========================================================

        # Probability-weighted expected MW

        test["expected_mw"] = (
            pd.to_numeric(
                test["total_mw"],
                errors="coerce"
            )
            *
            test["calibrated_probability"]
        )

        # =========================================================
        # KNOWN TEST OUTCOMES
        # =========================================================

        known = test[
            test["realized"].notna()
        ].copy()

        # =========================================================
        # CALIBRATION TABLE
        # =========================================================

        known["prob_bin"] = pd.cut(
            known["calibrated_probability"],
            bins=[
                0,
                0.05,
                0.10,
                0.20,
                0.30,
                0.50,
                1.00
            ],
            include_lowest=True
        )

        calibration_table = (
            known
            .groupby(
                "prob_bin",
                observed=True
            )
            .agg(
                projects=("realized", "count"),
                actual_rate=("realized", "mean"),
                predicted_rate=(
                    "calibrated_probability",
                    "mean"
                ),
                expected_mw=("expected_mw", "sum")
            )
        )

        print("\nCalibration Table:")
        print(calibration_table)

        # =========================================================
        # MODEL PERFORMANCE
        # =========================================================

        raw_auc = roc_auc_score(
            known["realized"],
            known["raw_probability"]
        )

        calibrated_brier = brier_score_loss(
            known["realized"],
            known["calibrated_probability"]
        )

        print("\nModel Performance:")
        print("ROC-AUC:", raw_auc)
        print("Brier Score:", calibrated_brier)

        print(
            "Average calibrated probability:",
            known["calibrated_probability"].mean()
        )

        print(
            "Actual realization rate:",
            known["realized"].mean()
        )

        print(
            "Expected MW:",
            known["expected_mw"].sum()
        )

        # =========================================================
        # CALIBRATION CURVE
        # =========================================================

        prob_true, prob_pred = calibration_curve(
            known["realized"],
            known["calibrated_probability"],
            n_bins=10,
            strategy="quantile"
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label="Model"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            linewidth=2,
            label="Perfect calibration"
        )

        plt.xlabel(
            "Predicted Probability",
            fontsize=12
        )

        plt.ylabel(
            "Observed Realization Rate",
            fontsize=12
        )

        plt.title(
            "Calibration Curve",
            fontsize=14
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()
        plt.show()

        # =========================================================
        # EXPECTED VS ACTUAL MW BY YEAR
        # =========================================================

        # IMPORTANT:
        #
        # Operational = known outcome
        # Withdrawn   = known outcome
        # Active      = unknown outcome
        #
        # Active projects are NOT treated as 0 MW.
        # They remain NaN for actual MW.

        plot_data = test.copy()

        # =========================================================
        # CONVERT VALUES TO NUMERIC
        # =========================================================

        plot_data["q_year"] = pd.to_numeric(
            plot_data["q_year"],
            errors="coerce"
        )

        plot_data["total_mw"] = pd.to_numeric(
            plot_data["total_mw"],
            errors="coerce"
        )

        plot_data["expected_mw"] = pd.to_numeric(
            plot_data["expected_mw"],
            errors="coerce"
        )

        # =========================================================
        # ACTUAL REALIZED MW
        # =========================================================

        # Start with NaN.
        #
        # This is critical because unknown outcomes
        # must NOT be treated as zero.

        plot_data["actual_mw"] = float("nan")

        # ---------------------------------------------------------
        # OPERATIONAL PROJECTS
        # ---------------------------------------------------------

        operational_mask = (
            plot_data["q_status"] == "operational"
        )

        plot_data.loc[
            operational_mask,
            "actual_mw"
        ] = plot_data.loc[
            operational_mask,
            "total_mw"
        ]

        # ---------------------------------------------------------
        # WITHDRAWN PROJECTS
        # ---------------------------------------------------------

        withdrawn_mask = (
            plot_data["q_status"] == "withdrawn"
        )

        plot_data.loc[
            withdrawn_mask,
            "actual_mw"
        ] = 0.0

        # ---------------------------------------------------------
        # ACTIVE PROJECTS
        # ---------------------------------------------------------

        # Active projects remain NaN.
        #
        # They are NOT 0.
        # They are NOT 1.
        #
        # Their actual outcome is unknown.

        # =========================================================
        # REMOVE INVALID YEARS / EXPECTED MW
        # =========================================================

        plot_data = plot_data.dropna(
            subset=[
                "q_year",
                "expected_mw"
            ]
        )

        # =========================================================
        # EXPECTED MW BY YEAR
        # =========================================================

        yearly_expected = (
            plot_data
            .groupby("q_year")
            .agg(
                expected_mw=("expected_mw", "sum")
            )
            .reset_index()
        )

        # =========================================================
        # ACTUAL MW BY YEAR
        # =========================================================

        # Only projects with known outcomes are included.
        #
        # Active projects have NaN actual_mw and therefore
        # do not contribute to actual MW.

        yearly_actual = (
            plot_data[
                plot_data["actual_mw"].notna()
            ]
            .groupby("q_year")
            .agg(
                actual_mw=("actual_mw", "sum"),
                known_projects=("actual_mw", "count")
            )
            .reset_index()
        )

        # =========================================================
        # COMBINE EXPECTED + ACTUAL
        # =========================================================

        yearly = pd.merge(
            yearly_expected,
            yearly_actual,
            on="q_year",
            how="left"
        )

        # =========================================================
        # PRINT RESULTS
        # =========================================================

        print("\nExpected vs Actual MW by Year:")

        print(
            yearly.to_string(
                index=False
            )
        )

        # =========================================================
        # GRAPH
        # =========================================================

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        # ---------------------------------------------------------
        # EXPECTED MW
        # ---------------------------------------------------------

        ax.plot(
            yearly["q_year"],
            yearly["expected_mw"],
            marker="o",
            markersize=7,
            linewidth=2.5,
            color="#4C78A8",
            label="Expected MW"
        )

        # ---------------------------------------------------------
        # ACTUAL REALIZED MW
        # ---------------------------------------------------------

        ax.plot(
            yearly["q_year"],
            yearly["actual_mw"],
            marker="o",
            markersize=7,
            linewidth=2.5,
            color="#F58518",
            label="Actual Realized MW"
        )

        # =========================================================
        # LABELS
        # =========================================================

        ax.set_xlabel(
            "Year",
            fontsize=13
        )

        ax.set_ylabel(
            "MW",
            fontsize=13
        )

        ax.set_title(
            "Expected vs. Actual Realized MW by Year",
            fontsize=16,
            pad=15
        )

        # =========================================================
        # X AXIS
        # =========================================================

        ax.set_xticks(
            yearly["q_year"]
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        # =========================================================
        # GRID
        # =========================================================

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.set_axisbelow(True)

        # =========================================================
        # LEGEND
        # =========================================================

        ax.legend(
            frameon=True
        )

        # =========================================================
        # LAYOUT
        # =========================================================

        plt.tight_layout()
        plt.show()

        # =========================================================
        # CURRENT ACTIVE PROJECTS FOR FORECASTING
        # =========================================================

        forecast = data[
            data["q_status"] == "active"
        ].copy()

        # =========================================================
        # RAW REALIZATION PROBABILITY
        # =========================================================

        forecast["raw_probability"] = (
            model_pipeline
            .predict_proba(
                forecast[features]
            )[:, 1]
        )

        # =========================================================
        # CALIBRATED PROBABILITY
        # =========================================================

        forecast["calibrated_probability"] = (
            iso.predict(
                forecast["raw_probability"]
            )
        )

        # =========================================================
        # EXPECTED MW
        # =========================================================

        forecast["expected_mw"] = (
            pd.to_numeric(
                forecast["total_mw"],
                errors="coerce"
            )
            *
            forecast["calibrated_probability"]
        )

        # =========================================================
        # RETURN
        # =========================================================

        return model_pipeline, test

    def forecast(self, MW, probability):

        # Probability-weighted expected load

        return MW * probability
