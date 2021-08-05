from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import Precision, Recall
from tensorflow.python.keras.layers import LSTM, Dense, Embedding, GRU
import numpy as np


def sequence(tweets, labels, max_words, max_len):
    tokenizer = Tokenizer(num_words=max_words, split=' ')
    tokenizer.fit_on_texts(tweets)
    x = tokenizer.texts_to_sequences(tweets)
    x = pad_sequences(x, max_len)
    y = np.array(labels)
    return x, y, tokenizer


# This GloVe embedding function was taken from the site:
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html and was also included as an example in
# TDT4310 lab 5 by https://github.com/micaelaustad/TDT4310_Hint/blob/main/Lab%205/code_snippets/DeepLearning.py
# The txt file can be found at: https://www.kaggle.com/bertcarremans/glovetwitter27b100dtxt, but is originally a part of
# https://nlp.stanford.edu/data/glove.twitter.27B.zip
def create_glove(tokenizer, max_len):
    word_index = tokenizer.word_index
    embeddings_index = {}
    f = open('data/Glove/glove.twitter.27B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return Embedding(len(word_index) + 1,
                     100,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False)


def build_lstm_model(embedding, output):
    model = Sequential()
    model.add(embedding)
    model.add(LSTM(128, dropout=0.5))
    model.add(Dense(output, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
    return model


def build_gru_model(embedding, outout):
    model = Sequential()
    model.add(embedding)
    model.add(GRU(32, dropout=0.5))
    model.add(Dense(outout, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
    return model


def train_model(model, ep, earlystopping, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=ep, batch_size=32, verbose=1, validation_data=(x_test, y_test),
              callbacks=earlystopping)
    return model


def save_model(model):
    model.save("my_model")


def evaluate_model(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test, verbose=1)
    f1_score = 2*(scores[2]*scores[3])/(scores[2] + scores[3])
    print(scores, f1_score)
    print("LOSS: %.2f" % (scores[0]))
    print("ACCURACY: %.2f" % (scores[1]))
    print("PRECISION: %.2f" % (scores[2]))
    print("RECALL: %.2f" % (scores[2]))
    print("F1-score: %.2f" % (f1_score))
    return scores


def predict(tweet, model, tokenizer, max_len):
    sequence = tokenizer.texts_to_sequences([tweet])
    padded = pad_sequences(sequence, maxlen=max_len)
    return np.around(model.predict(padded), decimals=0).argmax(axis=1)[0]