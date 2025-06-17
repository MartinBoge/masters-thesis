import polars as pl

from analysis.data.bronze.utils import BronzeConfig
from analysis.data.silver.utils import SilverConfig


def etl() -> None:
    etl_co2_emission()
    etl_forecast()


def etl_co2_emission() -> None:
    df = pl.read_parquet(f"{BronzeConfig.SAVE_DIR}/eds_co2_emission.parquet")

    df = df.select(
        pl.col("Minutes5UTC").str.to_datetime(time_zone="UTC").alias("minutes_5_utc"),
        pl.col("Minutes5DK").str.to_datetime(time_zone="Europe/Copenhagen", ambiguous="earliest").alias("minutes_5_dk"),
        pl.col("PriceArea").alias("price_area"),
        pl.col("CO2Emission").alias("co2_emission"),
    )

    df.write_parquet(f"{SilverConfig.SAVE_DIR}/eds_co2_emission.parquet")


def etl_forecast() -> None:
    df = pl.read_parquet(f"{BronzeConfig.SAVE_DIR}/eds_forecast.parquet")

    df = df.select(
        pl.col("HourUTC").str.to_datetime(time_zone="UTC").alias("hour_utc"),
        pl.col("HourDK").str.to_datetime(time_zone="Europe/Copenhagen", ambiguous="earliest").alias("hour_5_dk"),
        pl.col("PriceArea").alias("price_area"),
        pl.col("ForecastType").alias("forecast_type"),
        pl.col("ForecastDayAhead").alias("forecast_day_ahead"),
        pl.col("ForecastIntraday").alias("forecast_intraday"),
        pl.col("Forecast5Hour").alias("forecast_5_hour"),
        pl.col("ForecastCurrent").alias("forecast_current"),
        pl.col("Forecast1Hour").alias("forecast_1_hour"),
        pl.col("TimestampUTC").str.to_datetime(time_zone="UTC").alias("timestamp_utc"),
        pl.col("TimestampDK")
        .str.to_datetime(time_zone="Europe/Copenhagen", ambiguous="earliest")
        .alias("timestamp_dk"),
    )

    df.write_parquet(f"{SilverConfig.SAVE_DIR}/eds_forecast.parquet")


if __name__ == "__main__":
    etl()
