import pandas as pd
import requests
from datetime import datetime
import math


class GetFTXPriceData:
    """
    Makes API calls for the user and aggregates price data into pandas dataframe

    Parameters:
        interval: interval for the api calls; passed either in days, hours or minutes. i.e. '15m'
        date_start, date_stop: dates to make api calls for; should be passed as str in the form of: %d/%m/%y i.e. 17/07/22
        ticker: ticker for the api calls, passed as string in the form of: BASE_CURRENCY/QUOTE_CURRENCY i.e. LUNC/USD
    """

    def __init__(
        self, interval: str, date_start: str, date_stop: str, ticker: str = "LUNC/USD"
    ):
        self.interval, self.ticker = interval, ticker
        self.timestamp_start, self.timestamp_end = (
            self.get_timestamp_for_date(date_start),
            self.get_timestamp_for_date(date_stop),
        )

    def get_candlestick_data(self) -> pd.DataFrame:
        """
        Retrieves FTX Api price data
        """
        df = pd.DataFrame(
            requests.get(
                self.get_url_with_params(
                    self.timestamp_start, self.offset_timestamp(self.timestamp_start, 1)
                )
            ).json()["result"]
        )
        if self.get_api_calls_number() > 1:
            df = self.get_data_from_timestamps(df)
        df["date_time"] = pd.to_datetime(df.startTime, utc=True)
        return df.drop_duplicates()

    def get_data_from_timestamps(self, df) -> pd.DataFrame:
        """
        Loops through timestamps list and returns df with all the price data
        """
        for start, end in self.get_timestamps_list():
            df = (
                df.append(
                    pd.DataFrame(
                        requests.get(self.get_url_with_params(start, end)).json()[
                            "result"
                        ]
                    ),
                    ignore_index=True,
                )
                if end < self.timestamp_end
                else df.append(
                    pd.DataFrame(
                        requests.get(
                            self.get_url_with_params(start, self.timestamp_end)
                        ).json()["result"]
                    ),
                    ignore_index=True,
                )
            )
        return df

    def get_timestamps_list(self) -> list:
        """
        Retrieves all timestamps that need to be received from API (API data limits)
        """
        return [
            [
                self.offset_timestamp(self.timestamp_start, i + 1),
                self.offset_timestamp(self.timestamp_end, i + 2) - 1,
            ]
            for i in range(self.get_api_calls_number())
        ]

    def get_api_calls_number(self) -> int:
        """
        Calculate how many times API has to be requested for data
        """
        return math.ceil(
            (self.timestamp_end - self.timestamp_start)
            / (self.convert_interval() * 1500)
        )

    def offset_timestamp(self, timestamp_start, times=0) -> int:
        """
        Offsets timestamp for data retrieval
        """
        return timestamp_start + 1500 * self.convert_interval() * times

    @staticmethod
    def get_timestamp_for_date(date: str) -> int:
        """
        Returns timestamp for that date
        """
        return int(datetime.strptime(date, "%d/%m/%y").timestamp())

    def convert_interval(self) -> int:
        """
        Convert interval from human-readable (passed i.e. in hours) to match FTX api (needs to be passed in seconds)
        available intervals (from ftx api page):
        options: 15, 60, 300, 900, 3600, 14400, 86400, or any multiple of 86400 up to 30*86400
        that converts to: 15 sec, 1 min, 5 min, 15 min, 1 hr, 6 hrs, 1 day and multiples of days up to 30
        """
        num, interval = self.interval.lower().split(" ")
        if "d" in interval:
            if int(num) in range(1, 31):
                return int(num) * 86400
            else:
                raise Exception(
                    "Unsupported interval. For days, your interval should be an int the range of 1 to 30."
                )
        elif "h" in interval:
            if int(num) in (1, 6):
                return int(num) * 3600
            else:
                raise Exception(
                    "Unsupported interval. For hours, your interval should be either 1hrs or 6hrs"
                )
        elif "m" in interval:
            if int(num) in (1, 5, 15):
                return int(num) * 60
            else:
                raise Exception(
                    "Unsupported interval. For minutes, your interval should be either 1min, 5 mins or 15mins"
                )
        else:
            raise Exception(
                "Unsupported interval. Try d for days, h for hours or m for minutes"
            )

    def get_url_with_params(self, timestamp_start, timestamp_end) -> str:
        """
        Generates url for FTX api using starting and ending timestamp, interval and ticker
        """
        return f"""https://ftx.com/api/markets/{self.ticker}/candles?resolution={self.convert_interval()}&start_time={timestamp_start}&end_time={timestamp_end}"""


if __name__ == "__main__":
    prices = GetFTXPriceData("15 m", "01/05/22", "31/05/22").get_candlestick_data()
