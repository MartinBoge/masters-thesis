import datetime as dt
import os

from dotenv import load_dotenv

load_dotenv()


class BronzeConfig:
    FROM_DATETIME = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)
    TO_DATETIME = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dfs"
    EQ_API_KEY = os.environ["EQ_API_KEY"]
