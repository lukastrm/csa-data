import csv
import numpy
import re
from typing import List, Dict, Optional

EmojiSentimentDict = Dict[int, numpy.ndarray]

EMOJI_SENTIMENT_MAPPING_FILE = "emoji_sentiment_mapping.csv"

EMOJI_REGEX = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+"
)


def load_emoji_sentiment_mapping() -> EmojiSentimentDict:
    """
    Loads emoji sentiment mapping from an included a CSV file.

    :return: A dictionary of Unicode emoji characters (represented by their corresponding integer values) and 1d numpy
    arrays of length expressing the sentiment probabilities for negative, neutral and positive sentiment.
    """

    from importlib.resources import open_text
    from csadata import resources

    with open_text(resources, EMOJI_SENTIMENT_MAPPING_FILE) as file:
        reader = csv.reader(file)
        # Skip headline
        next(reader)

        emoji_sent_dict: EmojiSentimentDict = {}

        for line in reader:
            emoji = ord(line[0])
            num = int(line[2])
            negative = int(line[4])
            neutral = int(line[5])
            positive = int(line[6])

            # Calculate discrete sentiment probability distribution and assign it to numpy array
            emoji_sent_dict[emoji] = numpy.array([negative / num, neutral / num, positive / num])

        return emoji_sent_dict


def resolve_emoji_sentiments(emoji_data: List[str], emoji_sent_dict: Optional[EmojiSentimentDict] = None) -> \
        numpy.ndarray:
    """
    Resolves a list of emoij strings (data points) to a data array with the corresponding sentiment probability
    distributions for each emoji. A single emoji string is a string containing different single emojis. If multiple
    emojis are present in a single emoji string, the resulting probability distribution is the mean of each single
    distribution. If an emoji string does not contain any emojis or only contains emojis whose sentiment probability
    distribution is unknown, is is considered as neutral, i.e. p = [0 1 0] with p(negative) = p(positive) = 0 and
    p(neutral) = 1.

    :param emoji_data: The list of emoji strings which should be resolved.
    :param emoji_sent_dict: The emoji sentiment data as obtained from the load_emoji_sentiment_data function.
    :return:
    """

    if emoji_sent_dict is None:
        emoji_sent_dict = load_emoji_sentiment_mapping()

    data = numpy.zeros((len(emoji_data), 3))

    for i, s in enumerate(emoji_data):
        if len(s) == 0:
            # The data entry does not contain any emojis
            data[i, 1] = 1
            continue

        j = 0

        for emoji in s:
            sent = emoji_sent_dict.get(ord(emoji))

            if sent is not None:
                data[i] += sent
                j += 1

        if j > 0:
            # In case of different emojis the mean sentiment is calculated
            data[i] /= j
        else:
            # If for the given emojis no sentiment data is available it is considered as neutral
            data[i, 1] = 1

    return data


def extract_emojis(string: str):
    """
    Extracts all emojis out of a given string and returns a new string containing just the emojis.

    :param string: The string out of which the emojis should be extracted.
    :return: A string containing only the emojis or an empty string if no emojis were present.
    """
    return EMOJI_REGEX.sub("", string), "".join(EMOJI_REGEX.findall(string))
