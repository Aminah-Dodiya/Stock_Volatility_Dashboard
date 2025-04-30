import sqlite3
import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException
from config import settings
import logging

logger = logging.getLogger(__name__)

class AlphaVantageAPI:
    def __init__(self):
        self.api_key = settings.alpha_api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_daily(self, ticker, output_size="full"):
        """
        Get daily time series of an equity from AlphaVantage API.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the equity.
        output_size : str, optional
            "compact" (latest 100) or "full" (full history), default "full".

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with columns: 'open', 'high', 'low', 'close', 'volume'.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": output_size,
            "datatype": "json",
            "apikey": self.api_key
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            response_data = response.json()

            if "Time Series (Daily)" not in response_data:
                raise ValueError(f"Invalid API call or wrong ticker symbol '{ticker}'.")

            stock_data = response_data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(stock_data, orient="index", dtype=float)

            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df.columns = [col.split(". ")[1] for col in df.columns]

            return df

        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            raise
        except RequestException as req_err:
            logger.error(f"Request failed: {req_err}")
            raise
        except ValueError as val_err:
            logger.error(f"Validation error: {val_err}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

class SQLRepository:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def insert_table(self, table_name: str, records: pd.DataFrame, if_exists: str = "fail") -> dict:
        """
        Insert DataFrame into SQLite database as a table.
        """
        if not isinstance(records, pd.DataFrame):
            raise TypeError("Records must be a pandas DataFrame.")

        try:
            n_inserted = records.to_sql(
                name=table_name,
                con=self.connection,
                if_exists=if_exists,
                index=True
            )
            self.connection.commit()  # ensure write to disk
            logger.info(f"Inserted {n_inserted} records into '{table_name}'.")
            return {
                "transaction_successful": True,
                "records_inserted": n_inserted
            }
        except Exception as e:
            logger.error(f"Failed to insert records into '{table_name}': {e}")
            self.connection.rollback()
            return {
                "transaction_successful": False,
                "records_inserted": 0
            }

    def read_table(self, table_name: str, limit: int = None) -> pd.DataFrame:
        """
        Read table from database.
        """
        try:
            if limit:
                sql = f"SELECT * FROM '{table_name}' ORDER BY date DESC LIMIT {limit}"
            else:
                sql = f"SELECT * FROM '{table_name}'"

            df = pd.read_sql(
                sql=sql,
                con=self.connection,
                parse_dates=["date"],
                index_col="date"
            )

            logger.info(f"Read {len(df)} records from '{table_name}'.")
            return df
        except Exception as e:
            logger.error(f"Failed to read table '{table_name}': {e}")
            raise
