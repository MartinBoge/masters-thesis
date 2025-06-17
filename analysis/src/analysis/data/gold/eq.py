import polars as pl

from analysis.data.gold.utils import GoldConfig
from analysis.data.silver.utils import SilverConfig


def etl() -> None:
    files = [
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

    for file in files:
        df = pl.read_parquet(f"{SilverConfig.SAVE_DIR}/eq_{file}.parquet")
        df = df.rename({"date": "t"})
        df.write_parquet(f"{GoldConfig.SAVE_DIR}/eq_{file}.parquet")


if __name__ == "__main__":
    etl()
