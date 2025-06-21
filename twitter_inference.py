# Import core dependencies for model inference and data handling

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import LongTensor
from transformers import RobertaModel, RobertaTokenizer
from keras.preprocessing.sequence import pad_sequences as pad

# Set model and data processing hyperparameters

MAXLEN = 48              # Maximum tweet token length
OUTPUT_UNITS = 3         # Number of sentiment classes: positive, neutral, negative
LR = (4e-5, 1e-2)        # Learning rates for different network layers
DROP_RATE = 0.225        # Dropout rate for regularization
ROBERTA_UNITS = 768      # Feature size of RoBERTa model output

# Initialize tokenizer for RoBERTa base model
model = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model)

# Define Roberta model architecture for sentiment classification
class Roberta(nn.Module):
    """
    Custom Roberta-based classifier for tweet sentiment analysis.
    Adds a dropout and dense layer head on top of pre-trained RobertaModel.
    """
    def __init__(self):
        super(Roberta, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(DROP_RATE)
        self.roberta = RobertaModel.from_pretrained(model)
        self.dense = nn.Linear(ROBERTA_UNITS, OUTPUT_UNITS)
        
    def forward(self, inp, att):
        """
        Forward pass for tweet classification.
        Args:
            inp (Tensor): Input tweet token IDs [batch, MAXLEN]
            att (Tensor): Attention mask [batch, MAXLEN]
        Returns:
            Tensor: Softmax probabilities for each sentiment class
        """
        inp = inp.view(-1, MAXLEN)
        _, self.feat = self.roberta(inp, att)
        return self.softmax(self.dense(self.drop(self.feat)))

# Instantiate model and argument parser for CLI use
network = Roberta()
parser = argparse.ArgumentParser()

# Parse model path argument for loading trained weights
parser.add_argument('train_model_path')
args = parser.parse_args(); path = args.train_model_path

# Load the trained sentiment classification model
network.load_state_dict(torch.load(path + 'sentiment_model.pt'))

# Move model to GPU for inference and set to evaluation mode for efficiency
network = network.cuda().eval()

# Inference function to classify tweet sentiment
def predict_sentiment(tweet):
    """
    Predicts sentiment label for a single tweet string.
    Args:
        tweet (str): Raw tweet text
    Returns:
        str: Predicted sentiment label ('positive', 'neutral', or 'negative')
    """
    pg, tg = 'post', 'post'
    tweet_ids = tokenizer.encode(tweet.strip())
    sent = {0: 'positive', 1: 'neutral', 2: 'negative'}

    att_mask_idx = len(tweet_ids) - 1
    # Ensure token sequence starts with 0 (start token)
    if 0 not in tweet_ids: tweet_ids = 0 + tweet_ids
    # Pad and truncate tokens to MAXLEN
    tweet_ids = pad([tweet_ids], maxlen=MAXLEN, value=1, padding=pg, truncating=tg)

    # Build attention mask to match token structure
    att_mask = np.zeros(MAXLEN)
    att_mask[1:att_mask_idx] = 1
    att_mask = att_mask.reshape((1, -1))
    # Ensure final token is an end token
    if 2 not in tweet_ids: tweet_ids[-1], att_mask[-1] = 2, 0
    tweet_ids, att_mask = LongTensor(tweet_ids).cuda(), LongTensor(att_mask).cuda()
    # Run inference and return class label
    return sent[np.argmax(network.forward(tweet_ids, att_mask).detach().cpu().numpy())]

# Accept tweet text via command-line argument and print sentiment
parser.add_argument('tweet')
args = parser.parse_args()
print(predict_sentiment(args.tweet))