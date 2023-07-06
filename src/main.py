import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def data_preprocessing(path):
    # Read the JSONL file into a Pandas DataFrame
    df = pd.read_json(path, lines=True)
    df = df[['sentence1', 'sentence2', 'gold_label']]
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df

def vectorize_sentences(df, vectorizer):
    combined_sentences = df['sentence1'] + ' ' + df['sentence2']
    sentence_features = vectorizer.transform(combined_sentences)

    return sentence_features

def train_model_naive(X_train, y_train, X_test, y_test):
    
    print('\nNaive Bayes')
    model = GaussianNB()

    model.fit(X_train.toarray(), y_train)
    y_pred = model.predict(X_test.toarray())

    # Testing
    print(classification_report(y_test, y_pred))
    return model

def train_model_logistic(X_train, y_train, X_test, y_test):

    print('\nLogistic Regression')
    model = LogisticRegression()
    
    model.fit(X_train.toarray(), y_train)
    y_pred = model.predict(X_test.toarray())

    # Testing
    print(classification_report(y_test, y_pred))

def train_model_decision_tree(X_train, y_train, X_test, y_test):

    print('\nDescision Tree')
    model = DecisionTreeClassifier()
    model.fit(X_train.toarray(), y_train)

    y_pred = model.predict(X_test.toarray())

    # Testing
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":

    #training data
    preprocessed_data = data_preprocessing('snli_1.0/snli_1.0_train.jsonl')
    preprocessed_data = preprocessed_data[:5000]

    vectorizer = TfidfVectorizer()
    combined_sentences_training = preprocessed_data['sentence1'] + ' ' + preprocessed_data['sentence2']
    vectorizer.fit(combined_sentences_training)

    #vectorization training
    sentence_features_training = vectorize_sentences(preprocessed_data, vectorizer)
    labels_training = preprocessed_data['gold_label']

    #testing data
    test_data = data_preprocessing('snli_1.0/snli_1.0_test.jsonl')
    test_data = test_data[:5000]
    #vectorization testing
    combined_sentences_testing = test_data['sentence1'] + ' ' + test_data['sentence2']
    test_sentence_features = vectorizer.transform(combined_sentences_testing)
    test_labels = test_data['gold_label']

    #Naive Bayes
    model = train_model_naive(sentence_features_training, labels_training, test_sentence_features, test_labels)


    #manual passing
    '''
    sentence = 'Sphinx of black quartz, hear my vow'
    sentence_features = vectorizer.transform([sentence])
    temp = model.predict(sentence_features.toarray())
    print(temp)

    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(model, 'model.joblib')

    print('model and vectorizer stored')
    '''
    #Logistic Regression
    train_model_logistic(sentence_features_training, labels_training, test_sentence_features, test_labels)
    
    #Descision Tree
    train_model_decision_tree(sentence_features_training, labels_training, test_sentence_features, test_labels)