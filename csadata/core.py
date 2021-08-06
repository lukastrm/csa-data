from typing import List, Optional, Tuple, Dict

import numpy
import csadata.crypto as crypto
import csadata.twitter as twitter


INTERVAL_1MINUTE = (60, crypto.Client.KLINE_INTERVAL_1MINUTE)
INTERVAL_5MINUTE = (60 * 5, crypto.Client.KLINE_INTERVAL_5MINUTE)
INTERVAL_15MINUTE = (60 * 15, crypto.Client.KLINE_INTERVAL_15MINUTE)
INTERVAL_1HOUR = (60 * 60, crypto.Client.KLINE_INTERVAL_1HOUR)
INTERVAL_1DAY = (60 * 60 * 24, crypto.Client.KLINE_INTERVAL_1DAY)

IDX_BASE_TIME = 0
IDX_NEGATIVITY = 1
IDX_NEUTRALITY = 2
IDX_POSITIVITY = 3
IDX_NUM_TWEETS = 4

CRYPTO_PRICE_OFFSET = 5


def build_sentiment_data_set(start: int,
                             end: int,
                             interval: Tuple[int, str],
                             tweet_path: Optional[str] = None,
                             tweet_file_structure: Optional[twitter.TweetDataFileStructure] = None,
                             crypto_path: Optional[str] = None,
                             include_prices: List[str] = crypto.PRICES,
                             exclude_prices: Optional[List[str]] = None) -> Tuple[numpy.ndarray, Dict[str, int]]:
    """
    Returns a joint dataset with sentiment values for a given time interval that are based on the Tweets that have been
    posted within that interval paired with price data from different cryptocurrencies.

    :param start: The start time for the data set.
    :param end: The end time for the data set.
    :param interval: The time interval between two subsequent data points (i.e. the temporal resolution).
    :param tweet_path: The file path to the root directory of the Tweet data set.
    :param tweet_file_structure: The file structure of the Tweet data set.
    :param crypto_path: The file path to the cryptocurrency data set.
    :param include_prices: A list of price symbols whose related price data is included in the data set, defaults to all
    known price symbols.
    :param exclude_prices: A list of price symbols whose related price data is excluded from the data set.
    :return: A tuple of a 2d NumPy array with the Tweet sentiment and cryptocurrency price data as well as a dictionary
    containing the mapping of the included cryptocurrency symbols and their respective column indices in the data set.
    """
    tweet_reader_args = dict(
        path=tweet_path,
        start=start,
        end=end,
        file_structure=tweet_file_structure
    )

    interval_time = interval[0]
    interval_symbol = interval[1]

    with twitter.TweetCSVReader(scan_only=True, **tweet_reader_args) as reader:
        for num_tweets, _ in enumerate(reader):
            pass

    tweet_data = numpy.zeros((num_tweets + 1, 5))

    with twitter.TweetCSVReader(include_sentiment=True, **tweet_reader_args) as reader:
        for i, tweet in enumerate(reader):
            tweet_data[i, 0] = tweet.time
            tweet_data[i, 1:4] = tweet.sentiment
            tweet_data[i, 4] = max(tweet.like_count, 1)

    # Transform Tweet times into corresponding time interval base times
    tweet_data[:, 0] = tweet_data[:, 0] // interval_time * interval_time

    price_symbols = include_prices if exclude_prices is None else \
        [price for price in include_prices if price not in exclude_prices]
    num_prices = len(price_symbols)

    # Initialize the data array
    data = numpy.zeros((numpy.ceil((end - start) / interval_time).astype(int),
                        5 + num_prices * crypto.NUM_CANDLESTICK_FIELDS))

    for i, base_time in enumerate(range(start, end, interval_time)):
        # Extract Tweet data from the current time interval
        segment = tweet_data[tweet_data[:, 0] == base_time]

        # Weight Tweet sentiments
        likes = segment[:, 4]
        segment[:, 1:4] = segment[:, 1:4] * likes[:, None]

        # Assign values to data array and normalize time interval sentiment
        data[i, IDX_BASE_TIME] = base_time
        data[i, IDX_NEGATIVITY:IDX_POSITIVITY + 1] = segment[:, 1:4].sum(axis=0) / max(likes.sum(), 1)
        data[i, IDX_NUM_TWEETS] = segment.shape[0]

    dim_offset = CRYPTO_PRICE_OFFSET
    price_offsets = {}

    invalid_indices = []

    for price in price_symbols:
        price_data = crypto.load_crypto_price_data(price=price, interval=interval_symbol, start=start * 1000,
                                                   end=end * 1000, path=crypto_path)
        price_dict = {}

        for i in range(price_data.shape[0]):
            price_dict[int(price_data[i][crypto.IDX_OPEN_TIME] / 1000)] = i

        for i in range(data.shape[0]):
            candlestick_data_idx = price_dict.get(int(data[i][0]))

            if candlestick_data_idx is None:
                invalid_indices.append(i)
                print(f"Missing price data for index {i}, price {price} at base time {int(data[i][0])}")
                continue

            data[i][dim_offset:dim_offset+crypto.NUM_CANDLESTICK_FIELDS] = price_data[candlestick_data_idx]

        price_offsets[price] = dim_offset
        dim_offset += crypto.NUM_CANDLESTICK_FIELDS

    numpy.delete(data, numpy.unique(invalid_indices), axis=0)
    return data, price_offsets
