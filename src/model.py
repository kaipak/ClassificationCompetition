import sys
import os
import random
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim

class SarcasmDetector(object):

    def __init__(self, tokenizer_model: str = 'bert-base-uncased',
                 tokenizer_do_lc: bool = True,
                 model_criterion=nn.BCELoss()):
        """

        :param tokenizer_model:
        :param tokenizer_do_lc:
        :param model_criterion:
        """
        self.tokenizer_model = tokenizer_model
        self.tokenizer_do_lc = tokenizer_do_lc
        self.model_criterion = model_criterion

    