from analysis.data.bronze.eds import etl as bronze_eds_etl
from analysis.data.bronze.eq import etl as bronze_eq_etl
from analysis.data.gold.eds import etl as gold_eds_etl
from analysis.data.gold.eq import etl as gold_eq_etl
from analysis.data.gold.forecasting import etl as gold_forecasting_etl
from analysis.data.silver.eds import etl as silver_eds_etl
from analysis.data.silver.eq import etl as silver_eq_etl


def etl() -> None:
    bronze_eds_etl()
    bronze_eq_etl()
    silver_eds_etl()
    silver_eq_etl()
    gold_eds_etl()
    gold_eq_etl()
    gold_forecasting_etl()


if __name__ == "__main__":
    etl()
