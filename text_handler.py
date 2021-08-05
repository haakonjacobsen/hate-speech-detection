from emoji_handler import is_emoji, remove_emojis
from data_handler import get_slang_words
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import regex as re
import pandas as pd

# Global variables
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tweet_tokenizer = TweetTokenizer()
stop_words = set(stopwords.words('english'))
internet_slang = get_slang_words()


def word_exist(word):
    if not word.isalpha():
        return False
    if not wordnet.synsets(word):  # Checks if a word exist (faster than checking if word in nltk.words)
        return False
    return True


def get_correct_word(word):
    # TODO - Add logic to correct non aplhabetic characters to alphabetic (e.g. m8 -> mate, a$$hole -> asshole)
    return word


def clean_word(word, lem_over_stem):
    if is_emoji(word):
        return word
    # Remove hashtag and return hashtag text
    if word[0] == '#':
        return word[1:]
    if word in internet_slang:
        word = internet_slang[word]
        return lemmatizer.lemmatize(word) if lem_over_stem else stemmer.stem(word)
    elif word_exist(word):
        return lemmatizer.lemmatize(word) if lem_over_stem else stemmer.stem(word)  # Lemmatize or stem the word
    return word


def clean_tweet(text, use_emoji, lem_over_stem):
    # Remove URL, @User and contractions
    text = re.sub("@\\w+ *", "", text)
    text = re.sub(r'http\S+', '', text)
    text = decontract(text)
    # Convert and fix Emojis from html decimal
    text = ' '.join(tweet_tokenizer.tokenize(text))
    text = remove_emojis(text) if not use_emoji else text
    text = text.lower()
    tweet_tokens = tweet_tokenizer.tokenize(text)
    # Clean each word/token
    tweet_tokens = [clean_word(word, lem_over_stem) for word in tweet_tokens
                    if is_emoji(word) or ((word not in stop_words) and (len(word) > 2))]
    return ' '.join(tweet_tokens)


# Inspired by answer in: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
def decontract(sent):
    sent = re.sub(r"can\'t", "can not", sent)
    sent = re.sub(r"won\'t", "will not", sent)
    sent = re.sub(r"n\'t", " not", sent)
    sent = re.sub(r"\'re", " are", sent)
    sent = re.sub(r"\'s", " is", sent)
    sent = re.sub(r"\'d", " would", sent)
    sent = re.sub(r"\'ll", " will", sent)
    sent = re.sub(r"\'t", " not", sent)
    sent = re.sub(r"\'ve", " have", sent)
    sent = re.sub(r"\'m", " am", sent)
    return sent