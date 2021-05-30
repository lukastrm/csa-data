import os
import csv
from typing import Optional
from numpy import ndarray, loadtxt
from binance.client import Client


ENV_VAR_BINANCE_API_KEY = "BINANCE_API_KEY"

DEFAULT_DATA_DIRECTORY = "crypto_price_data"

IDX_OPEN_TIME = 0
IDX_OPEN = 1
IDX_HIGH = 2
IDX_LOW = 3
IDX_CLOSE = 4
IDX_VOLUME = 5
IDX_CLOSE_TIME = 6
IDX_QUOTE_ASSET_VOLUME = 7
IDX_NUMBER_OF_TRADES = 8
IDX_TAKER_BUY_BASE_ASSET_VOLUME = 9
IDX_TAKER_BUY_QUOTE_ASSET_VOLUME = 10

PRICE_BITCOIN_USDT = "BTCUSDT"
PRICE_ETHEREUM_USDT = "ETHUSDT"
PRICE_CARDANO_USDT = "ADAUSDT"
PRICE_BINANCECOIN_USDT = "BNBUSDT"
PRICE_XRP_USDT = "XRPUSDT"
PRICE_DOGECOIN_USDT = "DOGEUSDT"
PRICE_USDCOIN_USDT = "BUSDUSDT"
PRICE_POLKADOT_USDT = "DOTUSDT"
PRICE_UNISWAP_USDT = "UNIUSDT"
PRICE_INTERNET_COMPUTER_USDT = "ICPUSDT"
PRICE_BITCOIN_CASH_USDT = "BCHUSDT"
PRICE_CHAINLINK_USDT = "LINKUSDT"
PRICE_LITECOIN_USDT = "LTCUSDT"
PRICE_POLYGON_USDT = "MATICUSDT"
PRICE_STELLAR_USDT = "XLMUSDT"
PRICE_SOLANA_USDT = "SOLUSDT"
PRICE_ETHEREUM_CLASSIC_USDT = "ETCUSDT"
PRICE_VECHAIN_USDT = "VETUSDT"

PRICES = [
    PRICE_BITCOIN_USDT, PRICE_ETHEREUM_USDT, PRICE_CARDANO_USDT, PRICE_BINANCECOIN_USDT, PRICE_XRP_USDT,
    PRICE_DOGECOIN_USDT, PRICE_USDCOIN_USDT, PRICE_POLKADOT_USDT, PRICE_UNISWAP_USDT, PRICE_INTERNET_COMPUTER_USDT,
    PRICE_BITCOIN_CASH_USDT, PRICE_CHAINLINK_USDT, PRICE_LITECOIN_USDT, PRICE_POLYGON_USDT, PRICE_STELLAR_USDT,
    PRICE_SOLANA_USDT, PRICE_ETHEREUM_CLASSIC_USDT, PRICE_VECHAIN_USDT
]

INTERVAL_1HOUR = Client.KLINE_INTERVAL_1HOUR
INTERVAL_1DAY = Client.KLINE_INTERVAL_1DAY
INTERVAL_15MINUTE = Client.KLINE_INTERVAL_15MINUTE


def download_crypto_price_data(price: str, interval: str, start: Optional[int] = None, end: Optional[int] = None,
                               api_key: Optional[str] = None, path: str = None) -> None:
    """
    Uses the Binance (www.binance.com) API to download historical cryptocurrency price data and saves it to a CSV file.
    Each column of the generated CSV file corresponds to the single data fields in the Binance API response, more
    specifically the columns are open time, open price, high price, low price, close price, volume, close time,
    quote asset volume, number of trades, taker buy base asset volume, taker buy quote asset volume (the very last data
    column can be ignored).

    :param price: The price symbol.
    :param interval: The time interval between data points (time resolution).
    :param start: The start time for the data set.
    :param end: The end time for the data set.
    :param api_key: The Binance API key.
    :param path: The file path for the CSV file in which the downloaded data is saved. If no path is specified, a new
    file with a default name based on the other parameters is created in the current working directory.
    """

    # If no API key is provided as function argument, it has to be set as environment variable
    if api_key is None:
        api_key = os.environ[ENV_VAR_BINANCE_API_KEY]

    # Instantiate API client
    client = Client(api_key=api_key)

    if path is None:
        # If no path is specified, use default path in current working directory
        path = os.path.join(os.getcwd(), DEFAULT_DATA_DIRECTORY, _make_file_name(price, interval, start, end))

    # Check if path directories exist and create them if not
    dirname = os.path.dirname(path)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if os.path.isdir(path):
        path = os.path.join(path, _make_file_name(price, interval, start, end))

    with open(path, "a", newline="") as file:
        writer = csv.writer(file)

        while True:
            # Make API call
            data = client.get_historical_klines(symbol=price, interval=interval, start_str=start, end_str=end,
                                                limit=1000)

            # Break if no data was returned
            if data is None or len(data) == 0:
                break

            # Append price data to CSV file
            writer.writerows(data)

            # Examine the close time of the last data point. If it is less than the end time and the number of returned
            # data points matches the API limit then make a new call with a start time just after the close time of the
            # last data point.
            close_time = data[-1][6]

            if len(data) < 1000 or close_time >= end:
                break

            start = close_time + 1


def load_crypto_price_data(price: str, interval: str, start: Optional[int] = None, end: Optional[int] = None,
                           path: Optional[str] = None) -> ndarray:
    """
    Loads historical cryptocurrency price data from a CSV file that was downloaded by the download_crypto_price_data
    function.

    :param price: The price symbol.
    :param interval: The time interval between data points (time resolution).
    :param start: The start time for the data set.
    :param end: The end time for the data set.
    :param path: The file path for the CSV file in which the downloaded data is saved. If no path is specified, the
    method tries to find the file based on the other parameters in the current working directory.
    :return: A 2d NumPy array with the price data.
    """

    if path is None:
        # If no path is specified, use default path in current working directory
        path = os.path.join(os.getcwd(), DEFAULT_DATA_DIRECTORY, _make_file_name(price, interval, start, end))

    # Load data as NumPy array
    return loadtxt(path, delimiter=",", usecols=tuple(range(0, 11)))


def _make_file_name(price: str, interval: str, start: Optional[int], end: Optional[int]):
    return "_".join([price, interval, str(start), str(end)]) + ".csv"


class Candlestick:
    """
    Wrapper class for a single candlestick data point. The candlestick attributes of the underlying data point are
    accessed via computed attributes.
    """

    def __init__(self, data: ndarray):
        """
        :param data: A single candlestick data point as 1d NumPy array.
        """

        if len(data.shape) != 1 or data.shape[0] < 11:
            raise ValueError

        self._data: ndarray = data

    def __str__(self):
        return "Candlestick(open_time={}, open={}, high={}, low={}, close={}, volume={}, close_time={}, " \
               "quote_asset_volume={}, number_of_trades={}, taker_buy_base_asset_volume={}, " \
               "taker_buy_quote_asset_volume={})"\
            .format(self.open_time, self.open, self.high, self.low, self.close, self.volume, self.close_time,
                    self.quote_asset_volume, self.number_of_trades, self.taker_buy_base_asset_volume,
                    self.taker_buy_quote_asset_volume)

    @property
    def open_time(self):
        return self._data[IDX_OPEN_TIME]

    @property
    def open(self):
        return self._data[IDX_OPEN]

    @property
    def high(self):
        return self._data[IDX_HIGH]

    @property
    def low(self):
        return self._data[IDX_LOW]

    @property
    def close(self):
        return self._data[IDX_CLOSE]

    @property
    def volume(self):
        return self._data[IDX_VOLUME]

    @property
    def close_time(self):
        return self._data[IDX_CLOSE_TIME]

    @property
    def quote_asset_volume(self):
        return self._data[IDX_QUOTE_ASSET_VOLUME]

    @property
    def number_of_trades(self):
        return self._data[IDX_NUMBER_OF_TRADES]

    @property
    def taker_buy_base_asset_volume(self):
        return self._data[IDX_TAKER_BUY_BASE_ASSET_VOLUME]

    @property
    def taker_buy_quote_asset_volume(self):
        return self._data[IDX_TAKER_BUY_QUOTE_ASSET_VOLUME]
