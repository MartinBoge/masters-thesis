import datetime as dt
import math

import polars as pl

from analysis.data.gold.utils import GoldConfig


def etl() -> None:
    # Create empty model with timestamp col only
    hourly_datetimes = [
        GoldConfig.FROM_DATETIME + dt.timedelta(hours=i)
        for i in range(int((GoldConfig.TO_DATETIME - GoldConfig.FROM_DATETIME).total_seconds() / 3600))
    ]
    short_term_co2_forecasting_data = {"t": hourly_datetimes}
    df_model = pl.DataFrame(short_term_co2_forecasting_data)

    # Time dependent features
    df_model = df_model.with_columns(
        [
            ((pl.col("t") - pl.min("t")).dt.total_seconds() / 3600).cast(pl.Int64).alias("t_hours_since_start"),
            ((pl.col("t") - pl.min("t")).dt.total_seconds() / (24 * 3600)).cast(pl.Float64).alias("t_days_since_start"),
            ((pl.col("t") - pl.min("t")).dt.total_seconds() / (7 * 24 * 3600))
            .cast(pl.Float64)
            .alias("t_weeks_since_start"),
        ]
    )
    df_model = df_model.with_columns(
        [
            # Cyclic features for hour
            (pl.col("t").dt.hour() * (2 * math.pi / 24)).sin().alias("t_hour_sin"),
            (pl.col("t").dt.hour() * (2 * math.pi / 24)).cos().alias("t_hour_cos"),
            # Cyclic features for week
            (pl.col("t").dt.weekday() * (2 * math.pi / 7)).sin().alias("t_day_sin"),
            (pl.col("t").dt.weekday() * (2 * math.pi / 7)).cos().alias("t_day_cos"),
            # Cyclic features for month
            (pl.col("t").dt.month() * (2 * math.pi / 12)).sin().alias("t_month_sin"),
            (pl.col("t").dt.month() * (2 * math.pi / 12)).cos().alias("t_month_cos"),
            # Cyclic features for quarter
            (pl.col("t").dt.quarter() * (2 * math.pi / 4)).sin().alias("t_quarter_sin"),
            (pl.col("t").dt.quarter() * (2 * math.pi / 4)).cos().alias("t_quarter_cos"),
        ]
    )
    df_model = df_model.with_columns(
        # Linear features for year
        [
            pl.col("t").dt.year().alias("t_year"),
            (pl.col("t").dt.year() + (pl.col("t").dt.month() - 1) / 12).alias("t_year_month"),
        ]
    )

    # Join EQ data
    eq_files = [
        "operational_carbon_emission_kg",
        "consumption",
        "solar_photovoltaic_production",
        "wind_power_production_offshore",
        "wind_power_production_onshore",
        "residual_power_production_day_ahead",
        "price_spot_day_ahead_eur",
        "dk1_exchange_day_ahead_schedule_net_export",
        "temperature",
    ]

    for eq_file in eq_files:
        df_other = pl.read_parquet(f"{GoldConfig.SAVE_DIR}/eq_{eq_file}.parquet")
        df_model = df_model.join(df_other, on="t", how="left")

    df_model = df_model.with_columns(
        eq_operational_carbon_emission_t=pl.col("eq_operational_carbon_emission_kg") / 1000
    )

    # Finally sort, drop, and save
    df_model = df_model.sort("t", descending=False)
    df_model = df_model.drop("t", "eq_operational_carbon_emission_kg")
    df_model.write_parquet(f"{GoldConfig.SAVE_DIR}/forecasting_dataset.parquet")


if __name__ == "__main__":
    etl()
