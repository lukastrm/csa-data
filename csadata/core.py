from typing import List, Optional, Tuple, Dict

import numpy
import crypto
import twitter
import sentiment


_TWEET_SENTIMENT_TRANSFORMER_SHAPE = (5,)

INTERVAL_15MINUTE = (60 * 15, crypto.INTERVAL_15MINUTE)
INTERVAL_1HOUR = (60 * 60, crypto.INTERVAL_1HOUR)
INTERVAL_1DAY = (60 * 60 * 24, crypto.INTERVAL_1DAY)

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
    analyzer = sentiment.SentimentAnalyzer()

    tweet_transform_args = dict(
        path=tweet_path,
        start=start,
        end=end,
        shape=_TWEET_SENTIMENT_TRANSFORMER_SHAPE,
        transformer=lambda tweet: _tweet_sentiment_transformer(tweet, analyzer),
        file_structure=tweet_file_structure
    )

    interval_time = interval[0]
    interval_symbol = interval[1]

    # Transform the Tweets' text data into sentiment values
    tweet_data = twitter.transform(**tweet_transform_args)

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
        data[i, IDX_NEGATIVITY:IDX_POSITIVITY + 1] = segment[:, 1:4].sum(axis=0) / likes.sum()
        data[i, IDX_NUM_TWEETS] = segment.shape[0]

    dim_offset = CRYPTO_PRICE_OFFSET
    price_offsets = {}

    for price in price_symbols:
        price_data = crypto.load_crypto_price_data(price=price, interval=interval_symbol, start=start * 1000,
                                                   end=end * 1000, path=crypto_path)
        price_dict = {}

        for i in range(price_data.shape[0]):
            price_dict[int(price_data[i][crypto.IDX_OPEN_TIME] / 1000)] = i

        for i in range(data.shape[0]):
            candlestick_data_idx = price_dict.get(int(data[i][0]))

            if candlestick_data_idx is None:
                raise ValueError

            data[i][dim_offset:dim_offset+crypto.NUM_CANDLESTICK_FIELDS] = price_data[candlestick_data_idx]

        price_offsets[price] = dim_offset
        dim_offset += crypto.NUM_CANDLESTICK_FIELDS

    return data, price_offsets


def _tweet_sentiment_transformer(tweet: twitter.Tweet, analyzer: sentiment.SentimentAnalyzer):
    s = analyzer.classify_sentiment(tweet.text)
    return numpy.array([tweet.time, s[0], s[1], s[2], max(tweet.like_count, 1)])
