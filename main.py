from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from data_handler import (get_data, get_individual_tweets, undersample, plot_dataset_distr,
                          create_word_cloud, hs_distribution, plot_confusion_matrix)
from text_handler import clean_tweet
from models import (
    sequence, create_glove, build_lstm_model, build_gru_model, train_model,
    evaluate_model, save_model)
from sklearn.model_selection import train_test_split
from random import shuffle, seed
from keras import callbacks
from keras.layers import Embedding
from keras.models import load_model
import numpy as np

# Settings
USE_EMOJI = False
LEMMATIZE_WORDS = False    # If True lemmatize words, else use stemming
MAX_WORDS = 30000
MAX_LEN = 30
TEST_SIZE = 0.25

# -------- MAIN ----------
# 1. Import data
data = get_data()
# Extract three types of tweets: hate speech(hs), offensive(off) and not offensive(noff)
hs_tweets, off_tweets, noff_tweets = get_individual_tweets(data)
#plot_dataset_distr(hs_tweets, off_tweets, noff_tweets)

# 2. Make balanced datasets using undersampling/oversampling
hs_tweets_fix, off_tweets_fix, noff_tweets_fix = undersample(hs_tweets, off_tweets, noff_tweets)
#plot_dataset_distr(hs_tweets_fix, off_tweets_fix, noff_tweets_fix)
balanced_dataset = hs_tweets_fix + off_tweets_fix + noff_tweets_fix

seed(10)    # Used to shuffle the same way every time to detect improvements in model later
shuffle(balanced_dataset)

# 3. Clean data
clean_dataset = [(clean_tweet(tweet[0], USE_EMOJI, LEMMATIZE_WORDS), tweet[1])
                 for tweet in balanced_dataset]
clean_tweets, cleanlabels = zip(*clean_dataset)
#longest_string = max(clean_tweets, key=len)
#print(len(longest_string))

# 4. Vectorize and prepare data
tweets, labels = zip(*clean_dataset)
#  4.1 - Text to sequence
x, y, tokenizer = sequence(tweets, labels, MAX_WORDS, MAX_LEN)
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=34 + 35)
y_train_cat = to_categorical(y_train, 3)
y_test_cat = to_categorical(y_test, 3)

#  4.2 - Create GloVe and Keras embedding layer
glove_embedding = create_glove(tokenizer, MAX_LEN)
keras_embedding = Embedding(MAX_WORDS, 32, input_length=MAX_LEN)

# 5. Build models
#  5.1 - Create early stopping callback to find optimal epoch (minimize loss)
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
                                        mode ="min", patience = 5,
                                        restore_best_weights = True)
#  5.2 - Build and train LSTM model
lstm_model_glove = build_lstm_model(glove_embedding, 3)    # 3 output layers (hs = 0, off = 1, noff = 2)
lstm_model_keras = build_lstm_model(keras_embedding, 3)    # 3 output layers (hs = 0, off = 1, noff = 2)
train_model(lstm_model_glove, 100, earlystopping, x_train, y_train_cat, x_test, y_test_cat)
train_model(lstm_model_keras, 100, earlystopping, x_train, y_train_cat, x_test, y_test_cat)

#  5.3 - Build and train GRU model
gru_model_glove = build_gru_model(glove_embedding, 3)
gru_model_keras = build_gru_model(keras_embedding, 3)
train_model(gru_model_glove, 100, earlystopping, x_train, y_train_cat, x_test, y_test_cat)    # Saves this model, as it has best performance
train_model(gru_model_keras, 100, earlystopping, x_train, y_train_cat, x_test, y_test_cat)


# 6. Evaluate models
#   6.1 - Evaluate models
print('\nLSTM WITH GLOVE EMBEDDING')
stats_lstm_glove = evaluate_model(lstm_model_glove, x_test, y_test_cat)
print('\nLSTM KERAS EMBEDDING')
stats_lstm_keras = evaluate_model(lstm_model_keras, x_test, y_test_cat)
print('\nGRU WITH GLOVE EMBEDDING')
stats_gru_glove = evaluate_model(gru_model_glove, x_test, y_test_cat)
print('\nGRU WITH KERAS EMBEDDING')
stats_gru_keras = evaluate_model(gru_model_keras, x_test, y_test_cat)

#    6.2 - Save best model
save_model(lstm_model_glove)    # From manually checking the output above, we know it is: GloVe + LSTM (given settings)

saved_model = load_model("my_model")

#   6.2 - Plot confusion matrix
y_pred = np.argmax(saved_model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
plot_confusion_matrix(cm)

# 7. Extra experiments
#   7.1 - Display word distribution from word occuring in right and wrong hate speech prediction, respectivly
x_train, x_test, y_train, y_test = train_test_split(
        tweets, labels, test_size=TEST_SIZE, random_state=34 + 35)
words_right_pred, words_wrong_pred = hs_distribution(x_test, y_test, saved_model, tokenizer, MAX_LEN)
create_word_cloud(words_right_pred)
create_word_cloud(words_wrong_pred)
#   7.2 Testing on different dataset
# TODO - Test on a new dataset. (part og future work)