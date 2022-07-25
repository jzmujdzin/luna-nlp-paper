import ijson
import pandas as pd
import logging
from datetime import datetime
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger("tipper")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class TwitterDataHandler:
    """
    Handles snscrape scraped files from .json format,
    adds sentiment scores and filters tweets for particular
    language

    Parameters:
        file_tuple - a tuple of files (.json) containing all tweets
        lang - language to filter (i.e. 'en' for english)
    """

    def __init__(self, file_tuple, lang):
        self.file_tuple, self.lang = file_tuple, lang
        self.bert_classifier, self.vader_classifier = (
            pipeline("sentiment-analysis"),
            SentimentIntensityAnalyzer(),
        )
        self.twitter_df = pd.DataFrame()
        self.create_twitter_df()

    def create_twitter_df(self) -> None:
        """
        Merges files, adds sentiment and filters out language
        """
        logger.info(f"{datetime.now()} Starting to concatenate files")
        for file in self.file_tuple:
            self.concat_files(file)
        logger.info(f"{datetime.now()} Adding sentiment scores")
        self.add_sentiment_scores()
        self.twitter_df = self.twitter_df[self.twitter_df["lang"] == "en"]

    def concat_files(self, file) -> None:
        """
        Concatenates files, filters out unnecessary columns
        """
        f_name = file.split("\\")[-1]
        logger.info(f"""{datetime.now()} opening {f_name}""")
        with open(file, "r") as f:
            objects = ijson.items(f, "", multiple_values=True)
            self.twitter_df = self.twitter_df.append(
                pd.DataFrame(
                    (
                        [
                            pd.to_datetime(row["date"]),
                            row["content"],
                            row["replyCount"],
                            row["retweetCount"],
                            row["likeCount"],
                            row["quoteCount"],
                            row["lang"],
                        ]
                        for row in objects
                    ),
                    columns=self.get_columns_for_twitter_df(),
                ),
                ignore_index=True,
            )
        f_name = file.split("\\")[-1]
        logger.info(f"{datetime.now()} done with {f_name}")

    def add_sentiment_scores(self) -> None:
        """
        Adds sentiment scores for the df
        """
        self.twitter_df = (
            self.twitter_df.assign(
                bert_dict=lambda x: x["content"].apply(
                    lambda tweet: self.bert_classifier(tweet)
                ),
                vader_dict=lambda x: x["content"].apply(
                    lambda tweet: self.vader_classifier.polarity_scores(str(tweet))
                ),
            )
        ).assign(
            bert=lambda x: x["bert_dict"].apply(
                lambda s: (s[0]["score"]) * (1 if (s[0]["label"]) == "POSITIVE" else -1)
            ),
            vader=lambda x: x["vader_dict"].apply(lambda s: s["compound"]),
        )

    @staticmethod
    def get_columns_for_twitter_df() -> list:
        """
        Returns column names for twitter df
        """
        return [
            "date",
            "content",
            "replyCount",
            "retweetCount",
            "likeCount",
            "quoteCount",
            "lang",
        ]


if __name__ == "__main__":
    df = TwitterDataHandler(
        (
            r"E:\luna-nlp-paper\luna_01_08.json",
            r"E:\luna-nlp-paper\luna_09_16.json",
            r"E:\luna-nlp-paper\luna_17_24.json",
            r"E:\luna-nlp-paper\luna_25_31.json",
        ),
        "en",
    ).twitter_df
