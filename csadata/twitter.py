import os
import sys
import csv
from datetime import datetime, timezone
from typing import List, Iterator, Optional
from dateutil import parser
from searchtweets import gen_request_parameters, ResultStream

ENV_VAR_SEARCHTWEETS_BEARER_TOKEN = "SEARCHTWEETS_BEARER_TOKEN"

RECENT_SEARCH_ENDPOINT = "https://api.twitter.com/2/tweets/search/recent"
FULL_ARCHIVE_SEARCH_ENDPOINT = "https://api.twitter.com/2/tweets/search/all"

TWEET_DICT_FIELD_ID = "id"
TWEET_DICT_FIELD_CREATED_AT = "created_at"
TWEET_DICT_FIELD_AUTHOR_ID = "author_id"
TWEET_DICT_FIELD_TEXT = "text"
TWEET_DICT_FIELD_PUBLIC_METRICS = "public_metrics"
TWEET_DICT_FIELD_RETWEET_COUNT = "retweet_count"
TWEET_DICT_FIELD_REPLY_COUNT = "reply_count"
TWEET_DICT_FIELD_LIKE_COUNT = "like_count"
TWEET_DICT_FIELD_QUOTE_COUNT = "quote_count"


class Tweet:
    """
    A class holding relevant fields which characterize a single Tweet.
    """

    def __init__(self, tweet_dict):
        self.id = tweet_dict.get(TWEET_DICT_FIELD_ID)

        if self.id is None:
            raise TypeError

        self.time = int(parser.isoparse(tweet_dict[TWEET_DICT_FIELD_CREATED_AT]).timestamp())
        self.author_id = tweet_dict[TWEET_DICT_FIELD_AUTHOR_ID]
        self.like_count = tweet_dict[TWEET_DICT_FIELD_PUBLIC_METRICS][TWEET_DICT_FIELD_LIKE_COUNT]
        self.reply_count = tweet_dict[TWEET_DICT_FIELD_PUBLIC_METRICS][TWEET_DICT_FIELD_REPLY_COUNT]
        self.retweet_count = tweet_dict[TWEET_DICT_FIELD_PUBLIC_METRICS][TWEET_DICT_FIELD_RETWEET_COUNT]
        self.quote_count = tweet_dict[TWEET_DICT_FIELD_PUBLIC_METRICS][TWEET_DICT_FIELD_QUOTE_COUNT]
        self.text = tweet_dict[TWEET_DICT_FIELD_TEXT]
        self.emojis = None


class TwitterAPISearch(Iterator[Tweet]):
    """
    Iterator class for a single Twitter API search call.
    """

    def __init__(self, result_stream: ResultStream):
        self.result_stream = result_stream.stream()

    def __next__(self) -> Tweet:
        """
        :return: The next Tweet object in the search result stream.
        """
        while True:
            tweet_dict = next(self.result_stream)

            try:
                return Tweet(tweet_dict)
            except TypeError:
                continue


class TwitterAPI:
    """
    Twitter API wrapper class.
    """

    def __init__(self, bearer_token: Optional[str] = None):
        self.bearer_token = bearer_token if bearer_token is not None else os.environ[ENV_VAR_SEARCHTWEETS_BEARER_TOKEN]

    def search(self, endpoint: str, query: str) -> TwitterAPISearch:
        """
        Perform a call to the Twitter API search endpoint.
        :param endpoint: The endpoint to which the call is made, either recent tweets or full archive search.
        :param query: The Tweet search query.
        :return: A TwitterAPISearch iterator object which returns the search results as single Tweet objects.
        """

        search_args = {
            "bearer_token": self.bearer_token,
            "endpoint": endpoint
        }

        params = gen_request_parameters(query=query, results_per_call=100,
                                        tweet_fields="id,created_at,author_id,text,public_metrics")
        result_stream = ResultStream(request_parameters=params, max_requests=1, max_tweets=100, **search_args)
        return TwitterAPISearch(result_stream)


class TweetCSVWriter:
    def __init__(self, base_path: str, dir_intervals: List[int], file_interval: int):
        super().__init__()
        self.base_path: str = base_path

        prev = sys.maxsize * 2 + 1

        if dir_intervals is None or len(dir_intervals) == 0:
            self.dir_intervals = []
        else:
            for interval in dir_intervals:
                if interval >= prev:
                    raise ValueError("Directory time intervals must be descending")
                elif interval <= 0:
                    raise ValueError("Directory time intervals must be non-negative and non-zero")

                prev = interval

            self.dir_intervals: List[int] = dir_intervals

        if file_interval >= dir_intervals[-1]:
            raise ValueError("File time interval must be less than smallest directory time interval")
        elif file_interval <= 0:
            raise ValueError("File time interval must be non-negative and non-zero")

        self.file_interval: int = file_interval
        self.open_file = None
        self.open_file_base_time = -1
        self.csv_writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.open_file is not None:
            self.open_file.close()

    def write(self, tweet: Tweet):
        base_time = int(tweet.time // self.file_interval * self.file_interval)

        if self.open_file_base_time is not base_time:
            if self.open_file is not None:
                self.open_file.close()

            self.open_file_base_time = base_time

            path = self.base_path

            for dir_interval in self.dir_intervals:
                path += os.sep + datetime.fromtimestamp(int(tweet.time // dir_interval * dir_interval))\
                    .astimezone(timezone.utc).isoformat()

            path += os.sep + datetime.fromtimestamp(self.open_file_base_time).astimezone(timezone.utc).isoformat() + \
                    ".csv"
            dirname = os.path.dirname(path)

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            self.open_file = open(path, "a", newline="")
            self.csv_writer = csv.writer(self.open_file)

        self.csv_writer.writerow([tweet.time, tweet.id, tweet.author_id, tweet.like_count, tweet.reply_count,
                                  tweet.retweet_count, tweet.quote_count, tweet.text, tweet.emojis])
