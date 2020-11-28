import os, sys
import errno
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataPrep:

    def __init__(self, train_path: str = '../data/train.jsonl',
                 test_path: str = '../data/test.jsonl',
                 rm_cap: bool=False, rm_punc: bool=False):
        """

        :param train_path:
        :param test_path:
        :param rm_cap:
        :param rm_punc:
        """
        self.df = pd.read_json(train_path, lines=True)
        self.df['concat'] = (
                self.df['context'].str.join(" ") + " "
                + self.df['response']
        )
        self.df_test = pd.read_json(test_path, lines=True)
        self.df_test['concat'] = (
                self.df_test['context'].str.join(" ") + " "
                + self.df_test['response']
        )
        self.df_split = False # whether or not dataframes were split

    def train_test_split(self, train_size: float = 0.8, random_state: int=None,
                         shuffle: bool = True):
        """

        :param train_size:
        :param random_state:
        :param shuffle:
        :return:
        """
        self.df_train, self.df_validation = (
            train_test_split(self.df, train_size=train_size,
                             random_state=random_state,
                             shuffle=shuffle)
        )
        self.df_split = True
        print(f"{train_size} train ratio results in {self.df_train.shape[0]} "
              f"training observations and {self.df_validation.shape[0]} "
              f"validation observations.")

    def write_data(self, datapath: str = '../data/processed',
                   train_fname: str = 'train.json',
                   valid_fname: str = 'validation.json',
                   test_fname: str = 'test.json',
                   format: str = "json"):
        """
        :param datapath:
        :param train_fname:
        :param valid_fname:
        :param test_fname:
        :param format:
        :return:
        """
        rootpath = Path(datapath)

        try:
            os.makedirs(rootpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if self.df_split:
            self.df_train.to_json(rootpath / train_fname, orient='records',
                                  lines=True)
            self.df_validation.to_json(rootpath / valid_fname,
                                       orient='records', lines=True)

        self.df_test.to_json(rootpath / test_fname, orient='records',
                             lines=True)



