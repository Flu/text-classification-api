from telnetlib import X3PAD
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
import keras.models as models
import numpy as np

import train

def main(input: str, data_file: str, model_filename: str):
    model = models.load_model(model_filename)

    sentences = train.document_preprocessor([input])
    vectorizer = CountVectorizer(lowercase=False)
    vectorizer.fit(sentences)

    X = vectorizer.transform(sentences)
    from scipy.sparse import csr_matrix

    print(type(X))
    print(X.shape)


    X = csr_matrix((X.data, X.indices, np.pad(X, (0, 27554 - X.shape[1]), 'edge')))

    print(type(X))
    print(X.shape)
    print(X)

    result = model.predict(X)

    for i in range(len(result)):
        print(i, result[i])

# Call the main function with the input string and the name of the file read from the command line arguments
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python3 inference.py <input_string> <data_file> <model_file>')
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])