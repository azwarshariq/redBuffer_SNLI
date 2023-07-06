from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the vectorizer and model
vectorizer = joblib.load('models/vectorizer.joblib')
model = joblib.load('models/model.joblib')

print('Loaded model and vectorizer')
# Make prediction on a new sentence
sentence = 'Sphinx of black quartz, hear my vow'
sentence_features = vectorizer.transform([sentence])
temp = model.predict(sentence_features.toarray())
print(temp)
