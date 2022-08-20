from lib2to3.pgen2 import token
from turtle import shape
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import re
plt.style.use('ggplot')

def document_preprocessor(X):
    documents = []

    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
    
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
    
        # Converting to Lowercase
        document = document.lower()

        # Lemmatization and tokenization
        token = nltk.word_tokenize(document)
        tagged = nltk.pos_tag(token)

        final_document = ""
        for i in range(0, len(tagged)):
            final_document = final_document + " " + tagged[i][0] + "/" + tagged[i][1]
    
        documents.append(final_document)

    return documents

def plot_history(history):
    print(history.history.keys())
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def prepare_data(*filenames):
    y = np.ndarray([])
    sentences = []
    print(y.shape)
    for filename in filenames:
        print(type(filename))
        df = pd.read_csv(filename, names=['sentence', 'label'], sep='\t')

        print(type(df['label'].values))
        # Preprocess the data, tokenize it and serve it as a list of inputs ready to be vectorized
        sentences = sentences + document_preprocessor(df['sentence'].values)
        if y.shape == ():
            y = df['label'].values
        else:
            y = np.concatenate((y, df['label'].values))

    # Split the dataset into training set and test set
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

    # Initialize the vectorizer
    vectorizer = CountVectorizer(lowercase=False, max_features=3000)
    vectorizer.fit(sentences_train)

    # Changes the text tokenized documents to feature vectors
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, X_test, y_train, y_test

def create_model(shape):
    model = Sequential()
    model.add(layers.Dense(200, input_dim=shape, activation='relu'))
    model.add(layers.Dense(15, activation='sigmoid'))
    model.add(layers.Dense(units=3, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam', # adaptive moment estimation
                metrics=['accuracy'])
    return model

def train(*filenames: str):
    X_train, X_test, y_train, y_test = prepare_data(*filenames)

    model = create_model(X_train.shape[1])

    model.summary()

    history = model.fit(X_train, y_train,
                        epochs=30,
                        verbose=True,
                        validation_data=(X_test, y_test),
                        batch_size=100,
                        )

    model.save('model.h5')

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)

def main():
    train("data/data.csv", "data/jokes.csv")

if __name__ == "__main__":
    main()