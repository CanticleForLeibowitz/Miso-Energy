import pandas as pd
import numpy as np


class manipulate:

    def __init__(self, data_set1, data_set2):
        self.data_set1 = data_set1
        self.data_set2 = data_set2


    def drop_columns(self):
        self.data_set1.drop(
            [
                'Negotiated In Service Date',
                'Withdrawn Date',
                'POI Name',
                'Generating Facility'
            ],
            axis=1,
            inplace=True
        )


    def filter_data(self):

        active = self.data_set1[
            self.data_set1['Request Status'] == 'Active'
        ]

        done = self.data_set1[
            self.data_set1['Request Status'] == 'Done'
        ]

        withdrawn = self.data_set1[
            self.data_set1['Request Status'] == 'Withdrawn'
        ]

        legacy = self.data_set1[
            self.data_set1['Request Status'] == 'LEGACY: Done'
        ]

        print("Active Records:")
        print(len(active))

        print("Done Records:")
        print(len(done))

        print("Withdrawn Records:")
        print(len(withdrawn))

        print("Legacy Done:")
        print(len(legacy))


    def extract_dates(self):

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

        self.data_set2["planned_years"] = (
            self.data_set2["prop_date"]
            - self.data_set2["q_date"]
        ).dt.days / 365.25

        return self.data_set2


    def IA_phase_clean(self):

        return self.data_set2[
            "IA_phase_clean"
        ].value_counts(dropna=False)


    def log_mw(self):

        mw_cols = ["mw_1", "mw_2", "mw_3"]

        for col in mw_cols:
            self.data_set2[col] = pd.to_numeric(
                self.data_set2[col],
                errors="coerce"
            )

        # Calculate total MW
        self.data_set2["total_mw"] = (
            self.data_set2[mw_cols]
            .sum(axis=1, min_count=1)
        )

        # Flag negative MW
        self.data_set2["invalid_mw"] = (
            self.data_set2["total_mw"] < 0
        )

        print(
            "Negative MW records:",
            self.data_set2["invalid_mw"].sum()
        )

        # Negative MW isn't physically meaningful for this model
        self.data_set2.loc[
            self.data_set2["invalid_mw"],
            "total_mw"
        ] = np.nan

        # Calculate log MW only for valid values
        self.data_set2["log_mw"] = np.log1p(
            self.data_set2["total_mw"]
        )

        return self.data_set2