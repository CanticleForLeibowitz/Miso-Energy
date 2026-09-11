import pandas as pd
import numpy as np


class manipulate:

    def __init__(self, data_set1, data_set2):
        self.data_set1 = data_set1
        self.data_set2 = data_set2

    def drop_columns(self):

        # Remove columns that are not needed for the model
        self.data_set1.drop(
            [
                "Negotiated In Service Date",
                "Withdrawn Date",
                "POI Name",
                "Generating Facility"
            ],
            axis=1,
            inplace=True
        )

    def filter_data(self):

        # Display the number of projects in each status
        statuses = [
            "Active",
            "Done",
            "Withdrawn",
            "LEGACY: Done"
        ]

        for status in statuses:
            count = (
                self.data_set1["Request Status"] == status
            ).sum()

            print(f"{status} Records:")
            print(count)

    def extract_dates(self):

        # Convert date columns to datetime
        date_cols = [
            "q_date",
            "prop_date",
            "on_date",
            "wd_date",
            "ia_date"
        ]

        for col in date_cols:
            self.data_set2[col] = pd.to_datetime(
                self.data_set2[col],
                errors="coerce"
            )

        # Calculate the planned development time in years
        self.data_set2["planned_years"] = (
            self.data_set2["prop_date"]
            - self.data_set2["q_date"]
        ).dt.days / 365.25

        return self.data_set2

    def IA_phase_clean(self):

        # Show the number of projects in each interconnection phase
        return self.data_set2["IA_phase_clean"].value_counts(
            dropna=False
        )

    def log_mw(self):

        mw_cols = ["mw_1", "mw_2", "mw_3"]

        # Convert MW columns to numeric values
        for col in mw_cols:
            self.data_set2[col] = pd.to_numeric(
                self.data_set2[col],
                errors="coerce"
            )

        # Calculate total proposed MW
        self.data_set2["total_mw"] = (
            self.data_set2[mw_cols]
            .sum(axis=1, min_count=1)
        )

        # Identify invalid negative MW values
        self.data_set2["invalid_mw"] = (
            self.data_set2["total_mw"] < 0
        )

        print(
            "Negative MW records:",
            self.data_set2["invalid_mw"].sum()
        )

        # Treat negative MW as missing rather than using invalid values
        self.data_set2.loc[
            self.data_set2["invalid_mw"],
            "total_mw"
        ] = np.nan

        # Log-transform MW to reduce the effect of very large projects
        self.data_set2["log_mw"] = np.log1p(
            self.data_set2["total_mw"]
        )

        return self.data_set2