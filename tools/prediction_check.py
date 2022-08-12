import pandas as pd
import numpy as np
import logging
from datetime import datetime
from ftx_price_data import GetFTXPriceData
import itertools
import re
import string
import nltk

pd.options.mode.chained_assignment = None

logger = logging.getLogger("tipper")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

TWITTER_DATA_PATH = r"E:\luna-nlp-paper\tools\luna_tweets_sentiment_all.csv"


class PredictionCheck:
    def __init__(
        self,
        threshold: float,
        date_start: str,
        date_end: str,
        intervals: list,
        shifts: int,
        ngram_list: list,
        filter_for_ngrams: bool,
    ):
        self.threshold, self.date_start, self.date_end, self.intervals = (
            threshold,
            date_start,
            date_end,
            intervals,
        )
        self.shifts, self.ngram_list = shifts, ngram_list
        self.stopwords = nltk.corpus.stopwords.words("english")
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.twitter_data = (
            self.filter_df_for_ngrams(pd.read_csv(TWITTER_DATA_PATH), self.ngram_list)
            if filter_for_ngrams
            else pd.read_csv(TWITTER_DATA_PATH)
        )
        self.vader_df = self.twitter_data[abs(self.twitter_data.vader) > 0.25]
        self.bert_df = self.twitter_data[abs(self.twitter_data.bert) > 0.25]
        self.evaluation = self.merge_scores_for_methods()

    def calculate_interval_predictions(
        self, interval, df, score_column
    ) -> pd.DataFrame:
        """Retrieve predictions for certain interval"""
        return self.get_adj_prices(interval).merge(
            self.get_adj_twitter_df(df, score_column, interval),
            left_index=True,
            right_index=True,
        )

    def merge_scores_for_methods(self) -> pd.DataFrame:
        """Merge scores for both VADER and BERT methods into one DataFrame"""
        return pd.concat(
            [
                self.get_final_scores(self.vader_df, "vader"),
                self.get_final_scores(self.bert_df, "bert"),
            ]
        )

    def get_final_scores(self, df, score_column) -> pd.DataFrame:
        """Retrieve final scores for both methods, shifts and intervals"""
        logger.info(f"Getting final scores for {score_column}")
        return pd.DataFrame(
            [
                list(
                    itertools.chain.from_iterable(
                        [
                            [interval, shift, score_column],
                            self.calc_conf_matrix_stats(
                                interval, shift, df, score_column
                            ),
                        ]
                    )
                )
                for shift in range(self.shifts)
                for interval in self.intervals
            ],
            columns=self.get_final_scores_columns(),
        )

    @staticmethod
    def get_final_scores_columns() -> list:
        return [
            "interval",
            "lag",
            "method",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]

    def get_confusion_matrix(self, interval, lag, df, score_column) -> dict:
        """Retrieve confusion matrix for certain prediction"""
        logger.info(
            f"{datetime.now()} Calculating confusion matrix for {score_column},"
            f" interval: {interval} and lag: {lag}"
        )
        df = self.calculate_interval_predictions(interval, df, score_column)[
            ["real_pred", "price_up"]
        ]
        df = (
            df.assign(price_up=df["price_up"].shift(lag + 1)).drop(
                df[df["real_pred"] == 0].index
            )
        ).assign(
            tp=lambda x: x.apply(
                lambda u: 1
                if (u["real_pred"] == u["price_up"]) & (u["real_pred"] == 1)
                else 0,
                axis=1,
            ),
            tn=lambda x: x.apply(
                lambda u: 1
                if (u["real_pred"] == u["price_up"]) & (u["real_pred"] == -1)
                else 0,
                axis=1,
            ),
            fp=lambda x: x.apply(
                lambda u: 1 if (u["real_pred"] == 1) & (u["price_up"] == -1) else 0,
                axis=1,
            ),
            fn=lambda x: x.apply(
                lambda u: 1 if (u["real_pred"] == -1) & (u["price_up"] == 1) else 0,
                axis=1,
            ),
        )
        return {
            "tp": np.sum(df["tp"]),
            "fn": np.sum(df["fn"]),
            "fp": np.sum(df["fp"]),
            "tn": np.sum(df["tn"]),
        }

    def calc_conf_matrix_stats(self, interval, lag, df, score_column) -> list:
        """Retrieve Accuracy, Precision, Recall and F1 Score"""
        conf_matrix = self.get_confusion_matrix(interval, lag, df, score_column)
        return [
            self.calc_accuracy(conf_matrix),
            self.calc_precision(conf_matrix),
            self.calc_recall(conf_matrix),
            self.calc_f1_score(
                self.calc_recall(conf_matrix), self.calc_precision(conf_matrix)
            ),
        ]

    @staticmethod
    def calc_accuracy(conf_matrix) -> float:
        return (conf_matrix["tp"] + conf_matrix["tn"]) / sum(conf_matrix.values())

    @staticmethod
    def calc_recall(conf_matrix) -> float:
        return conf_matrix["tp"] / (conf_matrix["fn"] + conf_matrix["tp"])

    @staticmethod
    def calc_precision(conf_matrix) -> float:
        return conf_matrix["tp"] / (conf_matrix["fp"] + conf_matrix["tp"])

    @staticmethod
    def calc_f1_score(recall, precision) -> float:
        return (2 * recall * precision) / (recall + precision)

    @staticmethod
    def merge_dfs_with_shift(prices, twtr, shift, freq) -> pd.DataFrame:
        """Merge price and twitter DataFrames with certain shift"""
        return twtr.merge(
            prices.shift(shift, freq=freq),
            how="inner",
            left_index=True,
            right_index=True,
        )

    @staticmethod
    def interval_match(interval) -> str:
        """Match intervals (FTX and pandas use differently expressed intervals)"""
        return (
            re.findall(r"(\d+)(\w+?)", interval)[0][0]
            + " "
            + re.findall(r"(\d+)(\w+?)", interval)[0][1].lower()
        )

    def get_adj_prices(self, interval) -> pd.DataFrame:
        """Retrieve price data and whether the price went up"""
        return (
            GetFTXPriceData(
                self.interval_match(interval), self.date_start, self.date_end
            )
            .get_candlestick_data()
            .assign(
                price_up=lambda x: x.apply(
                    lambda col: 1 if (col["open"] < col["close"]) else -1, axis=1
                )
            )
            .set_index("date")
        )

    def get_adj_twitter_df(self, df, diff_col, interval) -> pd.DataFrame:
        """Manipulate twitter data: aggregate tweets for interval, return prediction if they met threshold"""
        df["date"] = pd.to_datetime(df["date"])
        df_twtr = df.groupby(pd.Grouper(key="date", freq=interval)).aggregate(np.mean)
        return (
            (
                df_twtr.assign(diff=df_twtr[diff_col] - df_twtr[diff_col].shift(1),)
            ).assign(
                meets_threshold=lambda x: x.apply(
                    lambda u: 1 if abs(u["diff"] / u[diff_col]) > self.threshold else 0,
                    axis=1,
                ),
                pred=lambda x: x["diff"].apply(lambda d: 1 if d > 0 else -1),
            )
        ).assign(
            real_pred=lambda x: x.apply(
                lambda u: (u["meets_threshold"] * u["pred"]), axis=1
            )
        )

    @staticmethod
    def check_for_ngrams(sentence, ngram) -> int:
        """Check if row contains ngrams"""
        return 1 if all(sentence.split().__contains__(wrd) for wrd in ngram) else 0

    def filter_df_for_ngrams(self, df, ngrams_list) -> pd.DataFrame:
        """Discards all rows that contain ngrams"""
        logger.info(f"Discarding unwanted ngrams")
        df = self.word_list_to_str(df).assign(
            ngram_count=lambda x: x.apply(
                lambda col: np.sum(
                    [
                        self.check_for_ngrams(col["processed_str"], ngram)
                        for ngram in ngrams_list
                    ]
                ),
                axis=1,
            )
        )
        return df[df["ngram_count"] == 0]

    def cleanup_data(self, tweet_content) -> list:
        """Lemmatize, remove links and punctuaction"""
        return [
            self.lemmatizer.lemmatize(word)
            for word in [
                word
                for word in re.split(
                    "\W+",
                    re.sub(
                        "[0-9]+",
                        "",
                        "".join(
                            [
                                char
                                for char in re.sub(
                                    "((www.[^s]+)|(https?://[^s]+))", " ", tweet_content
                                )
                                if char not in string.punctuation
                            ]
                        ),
                    ),
                )
                if word not in self.stopwords
            ]
        ]

    def word_list_to_str(self, df) -> pd.DataFrame:
        return (
            df.assign(
                processed_content=lambda x: x.apply(
                    lambda c: self.cleanup_data(c["content"]), axis=1
                )
            )
        ).assign(
            processed_str=lambda x: x.apply(
                lambda con: " ".join(
                    [str(word).lower() for word in con["processed_content"]]
                ),
                axis=1,
            )
        )


if __name__ == "__main__":
    sentiment_change_threshold = 0.05
    start_date = "01/05/22"
    end_date = "31/05/22"
    interval_list = ["5Min", "15Min", "1H", "4H"]
    lag_number = 6
    unwanted_ngrams = [
        ["price", "target", "next"],
        ["miss", "next", "move"],
        ["dont", "miss", "next"],
        ["cryptowhales", "saveluna", "luna"],
        ["lunar", "flying", "luna"],
        ["except", "btc", "eth", "beta"],
        ["mention", "updated", "every"],
        ["hotel", "del", "luna"],
    ]
    prediction_df = PredictionCheck(
        sentiment_change_threshold,
        start_date,
        end_date,
        interval_list,
        lag_number,
        unwanted_ngrams,
        filter_for_ngrams=False,
    ).evaluation
