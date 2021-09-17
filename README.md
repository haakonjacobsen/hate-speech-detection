# Hate speech detection - By Haakon Jacobsen
This project is a hate speech detection program developed as the main deliverable in the course TDT4310 - Intelligent Text Analytics and Language Understanding. The results from testing is documented in the attatched [paper](doc-paper.pdf). The program demonstrates how to detect hate speech, and is trained on the dataset of [Davidson et. al.](https://github.com/t-davidson/hate-speech-and-offensive-language). The program provides four Recurrent Neural Networks (RNN) using a combinations of LSTM/GRU and GloVe or Keras embedding. The program is configured to save the best performing model.

## Setup
To run the program you need to follow these steps:
  1. Download the pre-trained GloVe word vectors from [Standford](http://nlp.stanford.edu/data/glove.twitter.27B.zip) and put them in the datafolder like this: `data/glove/glove.twitter.27B.100d.txt`
  2. Install the dependencies: 
  ```
  pip install -r requirements.txt
  ```

## Run program
From the root folder, run:
```
python3 main.py
```

## Python files
The program contains 5 python files:

### main.py
This file is the excecutable file which runs the program. The file creates four neural networks, trains them,
and print out the score individually. Then it prints out a confusion matrix of the best performing model. It also prints out two word clouds
with the most common words in hate speech correctly classified as hate speech, and hate speech classified as offensive.
Be aware that the word clouds contain offensive and hateful words from hate speech tweets.

### data_handler.py
This file contains all logic concerning data handling such as importing, splitting and outputting graphics. 

### text_handler.py
This file contains all logic for cleaning textual data.

### emoji.py
This file contains logic for dealing with emojis.

### models.py
This file contains logic for dealing with neural networks. From building to evaluating models.

## Dataset
The dataset used can be found at [GitHub/t-davidson](https://github.com/t-davidson/hate-speech-and-offensive-language)
