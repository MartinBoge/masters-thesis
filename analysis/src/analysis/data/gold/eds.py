import polars as pl

from analysis.data.gold.utils import GoldConfig
from analysis.data.silver.utils import SilverConfig


def etl() -> None:
    etl_co2_emission()
    etl_forecast()


def etl_co2_emission() -> None:
    df = pl.read_parquet(f"{SilverConfig.SAVE_DIR}/eds_co2_emission.parquet")

    df = df.select(
        pl.col("minutes_5_utc").alias("t"),
        pl.col("co2_emission").alias("eds_co2_emission"),
    )

    df.write_parquet(f"{GoldConfig.SAVE_DIR}/eds_co2_emission.parquet")


def etl_forecast() -> None:
    df = pl.read_parquet(f"{SilverConfig.SAVE_DIR}/eds_forecast.parquet")

    df = df.select(
        pl.col("hour_utc").alias("t"),
        pl.col("forecast_type"),
        pl.col("forecast_day_ahead"),
    )

    forecast_types = df["forecast_type"].to_list()
    forecast_col_rename_mapping = {
        forecast_type: f"eds_forecast_mwh_{forecast_type.lower().replace(' ', '_')}" for forecast_type in forecast_types
    }

    df = df.pivot(
        on="forecast_type",
        index="t",
        values="forecast_day_ahead",
    )

    df = df.rename(forecast_col_rename_mapping)

    df.write_parquet(f"{GoldConfig.SAVE_DIR}/eds_forecast.parquet")


if __name__ == "__main__":
    etl()
