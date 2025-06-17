import datetime as dt
import os


class GoldConfig:
    FROM_DATETIME = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)
    TO_DATETIME = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dfs"
