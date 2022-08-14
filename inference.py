import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pickle
import train
import sys

def inference_for_document(message):
    X, _ = train.load_dataset()
    X.append(message)
    documents = train.document_preprocessor(X)
    X = train.document_features(documents, X)
    with open('text_classifier_1000', 'rb') as training_model:
        model = pickle.load(training_model)

    y_pred2 = model.predict(X[len(X)-1].reshape(1, -1))
    return y_pred2

if __name__ == '__main__':
    if len(sys.argv) > 1:
        result = inference_for_document(sys.argv[1])
        print(result)