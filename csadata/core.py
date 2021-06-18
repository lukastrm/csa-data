from typing import List, Optional, Tuple, Callable

import numpy
import crypto
import twitter
from datetime import datetime, timezone

_TWEET_SENTIMENT_TRANSFORMER_SHAPE = (4,)


def build_sentiment_data_set(start: int,
                             end: int,
                             interval: int,
                             crypto_interval: str,
                             output_path: Optional[str] = None,
                             twitter_path: Optional[str] = None,
                             crypto_path: Optional[str] = None,
                             tweet_file_structure: Optional[twitter.TweetDataFileStructure] = None,
                             include_prices: List[str] = crypto.PRICES,
                             exclude_prices: Optional[List[str]] = None) -> numpy.ndarray:
    # METHOD INCOMPLETE!
    tweet_transform_args = dict(
        path=twitter_path,
        start=start,
        end=end,
        shape=_TWEET_SENTIMENT_TRANSFORMER_SHAPE,
        transformer=_tweet_sentiment_transformer,
        file_structure=tweet_file_structure
    )

    tweet_data = twitter.transform(**tweet_transform_args)
    tweet_data[:, 0] = tweet_data[:, 0] // interval * interval

    base_time = start

    data = numpy.zeros((numpy.ceil((end - start) / interval).astype(int), 4))
    i = 0

    while base_time < end:
        segment = tweet_data[tweet_data[:, 0] == base_time]
        segment_sum = segment.sum(axis=0)
        data[i, 0:3] = segment_sum[1:]
        data[i, 3] = segment.shape[0]
        base_time += interval
        i += 1

    # TODO: Mean over tweets within time interval

    price_symbols = include_prices if exclude_prices is None else \
        [price for price in include_prices if price not in exclude_prices]
    num_prices = len(price_symbols)
    data = numpy.zeros((tweet_data.shape[0], tweet_data.shape[1] + 5 * num_prices))

    prices_dict = {}

    for price in price_symbols:
        crypto.load_crypto_price_data(price=price, interval=crypto_interval, start=start, end=end, path=crypto_path)


def _tweet_sentiment_transformer(tweet: twitter.Tweet):
    return numpy.array([tweet.time, 0, 0, 0]) # TODO: Replace with actual sentiment values


def _test_method():
    twitter_path = "/home/lukastrm/Documents/studies/crypto-sa-data/examples/data/"
    start = int(datetime(2021, 6, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())  # 06/01/2021 00:00:00
    end = int(datetime(2021, 6, 6, 23, 59, 59, tzinfo=timezone.utc).timestamp())  # 06/06/2021 23:59:59
    build_sentiment_data_set(start=start, end=end, interval=15 * 60, crypto_interval=crypto.INTERVAL_15MINUTE,
                             twitter_path=twitter_path)


if __name__ == "__main__":
    _test_method()