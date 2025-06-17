import json

import polars as pl
import pytz
import requests

from analysis.data.bronze.utils import BronzeConfig


def etl() -> None:
    base_url = "https://api.energidataservice.dk/dataset"

    param_start = BronzeConfig.FROM_DATETIME.astimezone(pytz.timezone("Europe/Copenhagen")).strftime("%Y-%m-%dT%H:%M")
    param_end = BronzeConfig.TO_DATETIME.astimezone(pytz.timezone("Europe/Copenhagen")).strftime("%Y-%m-%dT%H:%M")
    param_filter = json.dumps({"PriceArea": ["DK1"]})

    params = {
        "start": param_start,
        "end": param_end,
        "filter": param_filter,
    }

    datasets_and_files = {"co2emis": "co2_emission", "forecasts_hour": "forecast"}

    for dataset, file in datasets_and_files.items():
        res = requests.get(f"{base_url}/{dataset}", params)

        res.raise_for_status()

        data = res.json()["records"]

        df = pl.DataFrame(data)

        df.write_parquet(f"{BronzeConfig.SAVE_DIR}/eds_{file}.parquet")


if __name__ == "__main__":
    etl()
