from tabnanny import verbose
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pickle
from nltk.corpus import stopwords

# Load the dataset
def load_dataset():
    dialog_data = load_files(r"./txt_analysis/")
    X, y = dialog_data.data, dialog_data.target
    return X, y


# Preprocessing of documents
def document_preprocessor(X):
    documents = []
    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()

    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
    
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
    
        # Converting to Lowercase
        document = document.lower()
    
        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
    
        documents.append(document)
    return documents

# Converting words to features
def document_features(documents, X, max_features=7000, min_df=5, max_df=0.7):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(documents).toarray()

    from sklearn.feature_extraction.text import TfidfTransformer
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    return X

# Split data into train and test sets
def split_train_and_test(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

# Training to random forest algorithm
def train(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, verbose=1)
    classifier.fit(X_train, y_train)
    return classifier

def main():
    X, y = load_dataset()
    documents = document_preprocessor(X)
    X = document_features(documents, X)
    X_train, X_test, y_train, y_test = split_train_and_test(X, y)
    classifier = train(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Print statistics
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    with open('text_classifier_1000', 'wb') as picklefile:
        pickle.dump(classifier,picklefile)

if __name__ == '__main__':
    main()