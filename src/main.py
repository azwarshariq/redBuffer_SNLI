import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

def train_model(X_train, y_train, X_test, y_test):
    
    model = GaussianNB()
    model.fit(X_train.toarray(), y_train)

    y_pred = model.predict(X_test.toarray())

    return model, y_pred, y_test



if __name__ == "__main__":

    #training data
    preprocessed_data = data_preprocessing('snli_1.0/snli_1.0_train.jsonl')

    vectorizer = TfidfVectorizer()
    combined_sentences_training = preprocessed_data['sentence1'] + ' ' + preprocessed_data['sentence2']
    vectorizer.fit(combined_sentences_training)

    #vectorization training
    sentence_features_training = vectorize_sentences(preprocessed_data, vectorizer)
    labels_training = preprocessed_data['gold_label']

    #testing data
    test_data = data_preprocessing('snli_1.0/snli_1.0_test.jsonl')

    #vectorization testing
    combined_sentences_testing = test_data['sentence1'] + ' ' + test_data['sentence2']
    test_sentence_features = vectorizer.transform(combined_sentences_testing)
    test_labels = test_data['gold_label']

    # Train the model and make predictions
    model, y_pred, y_test = train_model(sentence_features_training, labels_training, test_sentence_features, test_labels)

    # Testing
    print(classification_report(y_test, y_pred))
