import os
import sys
import csv
from typing import List, Iterator, Optional, Callable, Tuple
from datetime import datetime, timezone
from dateutil import parser
from numpy import ndarray, array, zeros
from searchtweets import gen_request_parameters, ResultStream
from Levenshtein import distance

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

TWEET_TRANSFORMER_TEXT_LENGTH_SHAPE = (1,)
TWEET_TRANSFORMER_TEXT_LENGTH = lambda tweet: array((len(tweet.text)))

CONFIG_KEY_DIR_INTERVALS = "dir_intervals"
CONFIG_KEY_FILE_INTERVAL = "file_interval"

CONFIG = {
    CONFIG_KEY_DIR_INTERVALS: [60 * 60 * 24],
    CONFIG_KEY_FILE_INTERVAL: 60 * 15
}


class Tweet:
    """
    A class holding relevant fields which characterize a single Tweet.
    """

    def __init__(self, tweet_dict=None):
        if tweet_dict is None:
            self.id = None
            self.time = None
            self.author_id = None
            self.like_count = None
            self.reply_count = None
            self.retweet_count = None
            self.quote_count = None
            self.text = None
            return

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

    def __str__(self):
        return str(self.__dict__)


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


def _datetime_to_twitter_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


class TwitterAPI:
    """
    Twitter API wrapper class.
    """

    def __init__(self, bearer_token: Optional[str] = None):
        self.bearer_token = bearer_token if bearer_token is not None else os.environ[ENV_VAR_SEARCHTWEETS_BEARER_TOKEN]

    def search(self, endpoint: str, query: str, max_requests=None, max_tweets=None, start_time: Optional[int] = None,
               end_time: Optional[int] = None) -> TwitterAPISearch:
        """
        Perform a call to the Twitter API search endpoint.
        :param endpoint: The endpoint to which the call is made, either recent tweets or full archive search.
        :param query: The Tweet search query.
        :param max_requests: The maximum amount of requests to perform while searching.
        :param max_tweets: The maximum amount of Tweets
        :param start_time: The oldest timestamp from which Tweets are searched (in seconds since Epoch).
        :param end_time: The newest, most recent timestamp to which Tweets are searched (in seconds since Epoch).
        :return: A TwitterAPISearch iterator object which returns the search results as single Tweet objects.
        """

        search_args = {
            "bearer_token": self.bearer_token,
            "endpoint": endpoint
        }

        start_time_str = None if start_time is None else _datetime_to_twitter_iso(
            datetime.fromtimestamp(start_time, tz=timezone.utc))
        end_time_str = None if end_time is None else _datetime_to_twitter_iso(
            datetime.fromtimestamp(end_time, tz=timezone.utc))

        params = gen_request_parameters(query=query, results_per_call=100, start_time=start_time_str,
                                        end_time=end_time_str,
                                        tweet_fields="id,created_at,author_id,text,public_metrics")
        result_stream = ResultStream(request_parameters=params, max_requests=max_requests, max_tweets=max_tweets,
                                     **search_args)
        return TwitterAPISearch(result_stream)


def _nested_path(time: int, path: str, dir_intervals: List[int], file_base_time: int) -> str:
    path = path

    for dir_interval in dir_intervals:
        path += os.sep + datetime.fromtimestamp(int(time // dir_interval * dir_interval)).astimezone(timezone.utc)\
            .isoformat()

    path += os.sep + datetime.fromtimestamp(file_base_time).astimezone(timezone.utc).isoformat() + ".csv"
    return path


class TweetDataFileStructure:
    def __init__(self, dir_intervals: Optional[List[int]] = None, file_interval: Optional[int] = None):
        prev = sys.maxsize * 2 + 1

        if dir_intervals is not None:
            for interval in dir_intervals:
                if interval >= prev:
                    raise ValueError("Directory time intervals must be descending")
                elif interval <= 0:
                    raise ValueError("Directory time intervals must be non-negative and non-zero")

                prev = interval

        self.dir_intervals: List[int] = CONFIG[CONFIG_KEY_DIR_INTERVALS] if dir_intervals is None else dir_intervals

        if file_interval is not None:
            if len(self.dir_intervals) > 0 and file_interval >= self.dir_intervals[-1]:
                raise ValueError("File time interval must be less than smallest directory time interval")
            elif file_interval <= 0:
                raise ValueError("File time interval must be non-negative and non-zero")

        self.file_interval: int = CONFIG[CONFIG_KEY_FILE_INTERVAL] if file_interval is None else file_interval


class TweetCSVWriter:
    def __init__(self, path: str, file_structure: Optional[TweetDataFileStructure]):
        self.path: str = path
        self.fs: TweetDataFileStructure = TweetDataFileStructure() if file_structure is None else file_structure

        self.open_file = None
        self.open_file_base_time = -1
        self.csv_writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.open_file is not None:
            self.open_file.close()

    def write(self, tweet: Tweet):
        base_time = int(tweet.time // self.fs.file_interval * self.fs.file_interval)

        if self.open_file_base_time is not base_time:
            if self.open_file is not None:
                self.open_file.close()

            self.open_file_base_time = base_time

            path = _nested_path(tweet.time, self.path, self.fs.dir_intervals, base_time)
            dirname = os.path.dirname(path)

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            self.open_file = open(path, "a", newline="")
            self.csv_writer = csv.writer(self.open_file)

        self.csv_writer.writerow([tweet.time, tweet.id, tweet.author_id, tweet.like_count, tweet.reply_count,
                                  tweet.retweet_count, tweet.quote_count, tweet.text])


class TweetCSVReader(Iterator[Tweet]):
    def __init__(self, path: str, start: int, end: int, file_structure: Optional[TweetDataFileStructure] = None):
        self.path: str = path
        self.start: int = start
        self.end: int = end
        self.fs: TweetDataFileStructure = TweetDataFileStructure() if file_structure is None else file_structure

        self.open_file = None
        self.open_file_base_time = int(start // self.fs.file_interval * self.fs.file_interval)
        self.csv_reader = None

    def __enter__(self):
        self._close_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.open_file is not None:
            self.open_file.close()

    def __next__(self):
        tweet_data = None
        self._open_file()

        while tweet_data is None:
            if self.csv_reader is not None:
                try:
                    tweet_data = next(self.csv_reader)
                    break
                except StopIteration:
                    pass

            self._open_next_file()

        tweet = Tweet()
        tweet.time = int(tweet_data[0])
        tweet.id = tweet_data[1]
        tweet.author_id = tweet_data[2]
        tweet.like_count = int(tweet_data[3])
        tweet.reply_count = int(tweet_data[4])
        tweet.retweet_count = int(tweet_data[5])
        tweet.quote_count = int(tweet_data[6])
        tweet.text = tweet_data[7]
        return tweet

    def _open_next_file(self):
        self._close_file()
        self.open_file_base_time += self.fs.file_interval

        if self.open_file_base_time >= self.end:
            raise StopIteration

        self._open_file()

    def _close_file(self):
        if self.open_file is not None:
            self.open_file.close()
            self.open_file = None
            self.csv_reader = None

    def _open_file(self):
        if self.open_file is not None:
            return

        path = _nested_path(self.open_file_base_time, self.path, self.fs.dir_intervals, self.open_file_base_time)

        if os.path.isfile(path):
            self.open_file = open(path, "r", newline="")
            self.csv_reader = csv.reader(self.open_file)


class TweetDuplicateFilter:
    def __init__(self, batch_size: int = 100, dissimilarity_threshold: float = 0.1):
        self.batch_size: int = batch_size
        self.dissimilarity_threshold: float = dissimilarity_threshold
        self._tweets: List[Optional[Tweet]] = [None] * batch_size
        self._i: int = 0
        self._eliminated = False

    def feed(self, tweet: Tweet) -> bool:
        if self._eliminated:
            raise Exception

        self._tweets[self._i] = tweet
        self._i += 1

        if self._i == self.batch_size:
            self._eliminate()
            return True
        else:
            return False

    def pull(self) -> Optional[Tweet]:
        if not self._eliminated:
            raise Exception

        tweet = None

        while tweet is None:
            tweet = self._tweets[self._i]
            self._i += 1

            if self._i == self.batch_size:
                self._eliminated = False
                return tweet

        return tweet

    def reset(self) -> None:
        self._eliminated = False
        self._i = 0

    def _eliminate(self) -> None:
        for i in range(self._i):
            tweet_0 = self._tweets[i]

            if tweet_0 is None:
                continue

            for j in range(i + 1, self._i):
                tweet_1 = self._tweets[j]

                if tweet_1 is None:
                    continue

                tweet_text_0 = self._tweets[i].text
                tweet_text_1 = self._tweets[j].text
                dissimilarity = (distance(tweet_text_0, tweet_text_1) / max(len(tweet_text_0), len(tweet_text_1)))

                if dissimilarity < self.dissimilarity_threshold:
                    self._tweets[i] = None
                    self._tweets[j] = None
                    continue

        self._eliminated = True
        self._i = 0


class TweetDuplicateEliminator:
    def __init__(self, in_path: str, out_path: str, start: int, end: int, iterations: int = 1,
                 in_file_structure: Optional[TweetDataFileStructure] = None,
                 out_file_structure: Optional[TweetDataFileStructure] = None):
        self.reader: TweetCSVReader = TweetCSVReader(path=in_path, start=start, end=end,
                                                     file_structure=in_file_structure)
        self.writer: TweetCSVWriter = TweetCSVWriter(path=out_path, file_structure=out_file_structure)
        self.out_path: str = self.out_path
        self.iterations: int = iterations

    def eliminate(self) -> None:
        pass  # TODO: Add method


def transform(path: str, start: int, end: int, shape: Tuple[int], transformer: Callable[[Tweet], ndarray],
              file_structure: Optional[TweetDataFileStructure] = None) -> ndarray:
    reader_args = dict(path=path, start=start, end=end, file_structure=file_structure)

    with TweetCSVReader(**reader_args) as reader:
        for num_tweets, _ in enumerate(reader):
            pass

    data = zeros((num_tweets + 1,) + shape)

    with TweetCSVReader(**reader_args) as reader:
        for i, tweet in enumerate(reader):
            data[i] = transformer(tweet)

    return data
