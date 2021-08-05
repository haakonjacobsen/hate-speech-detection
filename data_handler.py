import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sn
from models import predict


def get_data():
    return pd.read_csv('data/set_2.csv')

def get_slang_words():
    # Constructed from: https://github.com/vedant-95/Twitter-Hate-Speech-Detection/blob/master/Data%20Cleaning.ipynb and
    # https://drive.google.com/file/d/19VGaLW5uapOv2TVTDJzgd0sOUR-243mF/view
    df_slang = pd.read_csv('data/internet_slang.csv')
    internet_slang = {slang: correct for slang, correct in zip(df_slang['slang'], df_slang['correct'])}
    return internet_slang

def get_individual_tweets(dataframe):
    hs_data = []
    off_data = []
    noff_data = []
    for tweet, label in zip(dataframe['tweet'], dataframe['class']):
        if label == 0:
            hs_data.append((tweet, label))
        elif label == 1:
            off_data.append((tweet, label))
        else:
            noff_data.append((tweet, label))
    return hs_data, off_data, noff_data


def undersample(hs_tweets, off_tweets, noff_tweets):
    threshold = min(len(hs_tweets), len(off_tweets), len(noff_tweets))
    return hs_tweets[:threshold], off_tweets[:threshold], noff_tweets[:threshold]


def plot_dataset_distr(hs_data, off_data, noff_data):
    # Extract label distirbution bar chart
    labels = ('Hate speech', 'Offensive', 'Neither')
    y_pos = np.arange(len(labels))
    label_occurance = [len(hs_data), len(off_data), len(noff_data)]
    plt.bar(labels, label_occurance, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Nr. of tweets')
    plt.title('Label distribution')
    plt.show()


def create_word_cloud(stats):
    wcloud = WordCloud(background_color='white').generate_from_frequencies(stats)
    plt.imshow(wcloud, interpolation="bilinear")
    plt.axis("off")
    (-0.5, 799, 399, -0.5)
    plt.show()

def plot_confusion_matrix(cm):
    df_cm = pd.DataFrame(cm, range(3), range(3))
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".0f") # font size
    plt.show()


def hs_distribution(x_test, y_test, model, tokenizer, MAX_LEN):
    words_hs_right = []
    words_hs_wrong = []
    for tweet, label in zip(x_test, y_test):
        if label == 0:
            predicted = predict(tweet, model, tokenizer, MAX_LEN)
            if predicted == label:
                words_hs_right += tweet.split(' ')
                print(f'RIGHT: predicted: {predicted}, was: {label} for text -> {tweet}')
            else:
                words_hs_wrong += tweet.split(' ')
                print(f'WRONG: predicted: {predicted}, was: {label} for text -> {tweet}')
    words_hs_right = dict(collections.Counter(' '.join(words_hs_right).split()))
    words_hs_wrong = dict(collections.Counter(' '.join(words_hs_wrong).split()))
    return words_hs_right, words_hs_wrong
