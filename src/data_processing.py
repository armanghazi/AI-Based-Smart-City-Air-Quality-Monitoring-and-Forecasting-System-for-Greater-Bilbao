# src/data_processing.py

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


NUMERIC_OBJECT_COLS = [
    "NO (µg/m3)",
    "CO (mg/m3)",
    "Benceno (µg/m3)",
    "Ortoxileno (µg/m3)",
    "Tolueno (µg/m3)",
    "M-P-XILENO (µg/m3)",
    "CO 8h (mg/m3)",
    "Etilbenceno (µg/m3)"
]

BAD_STATIONS_INITIAL = [
    "SAN_JULIAN",
    "ANDOAIN",
    "ANORGA"
]

BAD_STATIONS_FINAL = [
    "SANGRONIZ",
    "ABANTO",
    "ALONSOTEGI"
]


def load_data(filepath):
    """Load raw air quality dataset."""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def remove_invalid_stations(df):
    """Remove stations with poor quality data."""
    return df[
        ~df["station"].isin(BAD_STATIONS_INITIAL)
    ].copy()


def convert_numeric_columns(df):
    """Convert object columns containing numeric values."""
    df = df.copy()

    for col in NUMERIC_OBJECT_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

    return df


def filter_recent_period(df, start_year=2015):
    """Keep data from selected year onward."""
    return df[df["Date"].dt.year >= start_year].copy()


def drop_high_missing_columns(df, threshold=60):
    """
    Drop columns whose missing percentage exceeds threshold.
    """
    missing_percent = df.isna().mean() * 100

    cols_to_drop = missing_percent[
        missing_percent > threshold
    ].index

    return df.drop(columns=cols_to_drop)


def drop_redundant_pollutants(df):
    """
    Remove highly correlated variables.
    """

    redundant_cols = [
        "NO (µg/m3)",
        "NOX (µg/m3)"
    ]

    existing = [
        col for col in redundant_cols
        if col in df.columns
    ]

    return df.drop(columns=existing)


def remove_low_quality_stations(df):
    """Remove stations with excessive missing values."""
    return df[
        ~df["station"].isin(BAD_STATIONS_FINAL)
    ].copy()


def impute_missing_values(df):
    """
    Perform Iterative Imputation (MICE)
    station by station.
    """

    df = df.copy()

    num_cols = df.select_dtypes(
        include="number"
    ).columns

    imputer = IterativeImputer(
        max_iter=20,
        random_state=42
    )

    imputed_dfs = []

    for station in df["station"].unique():

        sub = df[
            df["station"] == station
        ].copy()

        non_empty_cols = [
            c for c in num_cols
            if not sub[c].isna().all()
        ]

        sub[non_empty_cols] = imputer.fit_transform(
            sub[non_empty_cols]
        )

        imputed_dfs.append(sub)

    return pd.concat(imputed_dfs).sort_index()


def save_parquet(df, output_path):
    """Save processed dataset."""
    df.to_parquet(output_path, index=False)


def run_pipeline(input_file, output_file):
    """
    Complete preprocessing workflow.
    """

    print("Loading data...")
    df = load_data(input_file)

    print("Removing invalid stations...")
    df = remove_invalid_stations(df)

    print("Converting numeric columns...")
    df = convert_numeric_columns(df)

    print("Filtering years >= 2015...")
    df = filter_recent_period(df)

    print("Dropping high missing columns...")
    df = drop_high_missing_columns(df)

    print("Removing redundant pollutants...")
    df = drop_redundant_pollutants(df)

    print("Removing low quality stations...")
    df = remove_low_quality_stations(df)

    print("Imputing missing values...")
    df = impute_missing_values(df)

    print("Saving parquet file...")
    save_parquet(df, output_file)

    print("Pipeline completed.")

    return df


if __name__ == "__main__":

    INPUT_FILE = (
        "data/processed/"
        "air_quality_bilbao_2012_2026.csv"
    )

    OUTPUT_FILE = (
        "data/processed/"
        "final_air_quality.parquet"
    )

    run_pipeline(
        INPUT_FILE,
        OUTPUT_FILE
    )