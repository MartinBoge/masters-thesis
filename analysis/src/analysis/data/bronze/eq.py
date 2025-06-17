from energyquantified import EnergyQuantified
from energyquantified.metadata import Aggregation
from energyquantified.time import Frequency

from analysis.data.bronze.utils import BronzeConfig


def etl() -> None:
    eq = EnergyQuantified(api_key=BronzeConfig.EQ_API_KEY)

    curves_and_files = {
        # Carbon emission
        "DK1 Carbon Operational Emission Power Production kgCO2eq 15min Synthetic": "operational_carbon_emission_kg",
        # Consumption
        "DK1 Consumption MWh/h 15min Backcast": "consumption",
        # Production
        "DK1 Solar Photovoltaic Production MWh/h 15min Backcast": "solar_photovoltaic_production",
        "DK1 Wind Power Production Offshore MWh/h 15min Backcast": "wind_power_production_offshore",
        "DK1 Wind Power Production Onshore MWh/h 15min Backcast": "wind_power_production_onshore",
        "DK1 Residual Production Day-Ahead MWh/h H Backcast": "residual_power_production_day_ahead",
        # Market
        "DK1 Price Spot Day-Ahead EUR/MWh H Backcast": "price_spot_day_ahead_eur",
        "DK1 Exchange Day-Ahead Schedule Net Export MWh/h H Backcast": "dk1_exchange_day_ahead_schedule_net_export",
        # Weather
        "DK1 Consumption Temperature °C 15min Synthetic": "temperature",
    }

    for curve, file in curves_and_files.items():
        timeseries = eq.timeseries.load(
            curve,
            begin=BronzeConfig.FROM_DATETIME,
            end=BronzeConfig.TO_DATETIME,
            time_zone="UTC",
            frequency=Frequency.PT1H,
            aggregation=Aggregation.MEAN,
        )

        df = timeseries.to_polars_dataframe(name=f"eq_{file}")

        if df.shape[0] == 0:
            print(f"No rows/access for curve: '{curve}'")
            continue

        df.write_parquet(f"{BronzeConfig.SAVE_DIR}/eq_{file}.parquet")


if __name__ == "__main__":
    etl()
