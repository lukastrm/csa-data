# crypto-sa - Data Retrieval

This repository holds the contents for the _Neural Information Processing_ project **Sentimental 
Analysis for Predicting Cryptocurrency Markets (crypto-sa)** at the Technical University of Berlin 
in 2021.

## Module Contents

### Package `csadata.twitter`

tbd.

### Package `csadata.crypto`

Use the `download_crypto_price_data(price, interval, start, end, api_key, path)` method to download
historical cryptocurrency price data via the [Binance API](https://binance-docs.github.io/apidocs/).
Please refer to the API specification in order to determine the desire price symbol, which is passed
as `price` parameter. For the interval parameter either specify on of the packages `INTERVAL_HOURLY`
or `INTERVAL_DAILY` constants or refer to other valid interval enumeration values from the API
specification. The `start` and `end` parameters refer to the start and end times of the whole search
interval and are given in UNIX timestamps (milliseconds since Epoch). If omitted, both parameters
default to `None` and search results are limited to 1000 data points. In order to make an API request
with this method, the `api_key` parameter or the environment variable as specified by the package
constant `ENV_VAR_BINANCE_API_KEY` has to be set to a valid API key. This key can be retrieved via a
Binance account. Please refer to the API specification for further information on authorization.

The method downloads price data and writes it into a CSV file at a file path as specified with the
`path` parameter. If no `path` parameter is specified, the data is stored in the current working
directory.

Use the `load_crypto_price_data(price, interval, start, end, path)` method to load previously
downloaded historical price data into a Numpy array.

### Package `csadata.emoji`

Use the `resolve_emoji_sentiments(emoji_data, emoji_sent_dict)` to resolve a list of N emoji strings 
to an N-by-3 Numpy array in which each row refers to a single emoji string and the 3 columns represent
the sentiment values for negativity, neutrality and positivity. If the `emoji_sent_dict` parameter is
not present, this method internally calls the `load_emoji_sentiment_mapping()` method, which loads
the corresponding emoji sentiment mapping from the module resources. This mapping is a dictionary of
integer values corresponding to the unicode code of the specified emoji and unidimensional Numpy arrays
of length 3 corresponding to the sentiment values. By calling the `load_emoji_sentiment_mapping()` 
method separately, it is possible to modify or extend this mapping programmatically before passing the
mapping dictionary as `emoji_sent_dict` parameter to the `resolve_emoji_sentiments(...)` method.

The default emoji sentiment mapping uses the data from the 
[Emoji Sentiment Ranking](http://kt.ijs.si/data/Emoji_sentiment_ranking/).

The `extract_emojis(string)` method is a helper method to extract an emoji string out of another given
string (which may also contain other characters).
