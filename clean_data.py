import pandas as pd
import numpy as np
from typing import Protocol

def load_data_from_csv(csv_path: str = "combined_data.csv") -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(csv_path)


class DataCleaner(Protocol):
    def run(self) -> pd.DataFrame: ...


class FormartDateColumns:
    def __init__(self, date_columns: list[str]):
        self.date_columns = date_columns

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format specified columns as datetime."""
        for col in self.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df


class DecimalPrecision:
    def __init__(self, date_col: str, precision: int = 2, decimal_columns: list[str] | None = None):
        self._decimal_columns = decimal_columns if decimal_columns is not None else []
        self.precision = precision
        self.date_col = date_col

    def _get_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Get numeric columns from the DataFrame."""
        return df.select_dtypes(include="number").columns.tolist()

    def _set_default_decimal_columns(self, df: pd.DataFrame):
        """Set default decimal columns if none are specified."""
        if not self._decimal_columns:
            self.decimal_columns = [
                col 
                for col in self._get_numeric_columns(df) 
                if col not in [self.date_col]
            ]
            return
        self.decimal_columns = self._decimal_columns

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round specified columns to the given decimal precision."""
        self._set_default_decimal_columns(df)
        for col in self.decimal_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(self.precision)
        return df


class FillMissingValues:
    def __init__(self, missing_values: list[object] | None = None, window: int = 3, skip_columns: list[str] | None = None):
        self.missing_values = missing_values if missing_values is not None else []
        self.window = window
        self.skip_columns = skip_columns if skip_columns is not None else []

    def _mark_custom_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace user-defined placeholders with missing values."""
        if not self.missing_values:
            return df

        for value in self.missing_values:
            if pd.isna(value):
                continue

            if isinstance(value, str):
                string_cols = df.select_dtypes(include=["object", "string"]).columns
                target = value.strip().lower()

                for col in string_cols:
                    normalized = df[col].astype("string").str.strip().str.lower()
                    df.loc[normalized == target, col] = pd.NA
                continue

            df = df.mask(df.eq(value), pd.NA)

        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the DataFrame."""
        df = self._mark_custom_missing_values(df)

        window = self.window
        numeric_cols = df.select_dtypes(include="number").columns
        numeric_cols = [col for col in numeric_cols if col not in self.skip_columns]

        for col in numeric_cols:
            series = df[col]
            moving_avg = series.rolling(window=window, min_periods=1).mean()
            df[col] = series.fillna(moving_avg)

            if df[col].isna().any():
                df[col] = df[col].fillna(series.mean())

        return df


class DropConstantColumns:
    def __init__(self, target_col: str, date_col: str):
        self.target_col = target_col
        self.date_col = date_col
        self._variance_threshold = 1e-5

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with near-zero variance."""
        constant_cols = []

        for col in df.columns:
            if col in (self.target_col, self.date_col):
                continue

            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                if series.var(skipna=True) < self._variance_threshold:
                    constant_cols.append(col)
            else:
                if series.nunique(dropna=True) <= 1:
                    constant_cols.append(col)

        return df.drop(columns=constant_cols)


class CleanData:
    def __init__(self, df: pd.DataFrame, target_col: str, date_col: str, cleaners: list[DataCleaner] = None):
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.cleaners = cleaners if cleaners is not None else []

    def clean(self) -> pd.DataFrame:
        """Run all data cleaning steps."""
        df = self.df.copy()
        for cleaner in self.cleaners:
            df = cleaner.run(df)
        self.df = df
        return self.df

    def get_columns(self) -> list:
        """Get the list of columns in the DataFrame."""
        return self.df.columns.tolist()


def main():
    # Load data from CSV
    df = load_data_from_csv("data/combined_data.csv")
    data_cleaner = CleanData(
        df=df, 
        target_col="sales", 
        date_col="month",
        cleaners=[
            DropConstantColumns(target_col="sales", date_col="month"),
            FillMissingValues(missing_values=[np.nan, 0, "zero"], skip_columns=["month", "fy"]),
            FormartDateColumns(date_columns=["month"]),
            DecimalPrecision(date_col="month")
        ]
    )
    cleaned_df = data_cleaner.clean()

    cleaned_df.to_csv("data/cleaned_combined_data.csv", index=False)

    print(f"Data cleaned: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")


if __name__ == "__main__":
    main()

    