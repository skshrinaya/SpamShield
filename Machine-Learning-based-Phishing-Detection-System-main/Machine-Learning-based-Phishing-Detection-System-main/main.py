import re
from urllib.parse import urlparse
import pickle
from flask import Flask, request, jsonify, render_template
import joblib
from Feature_Extract import extract_features
from API import get_prediction
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from flask_cors import CORS  # Import Flask-CORS

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "chrome-extension://gecnpejaonkefcdmbjjdbdllejoppjdo"}})
CORS(app)  # Initialize Flask-CORS

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model1 = pickle.load(open('model.pkl', 'rb'))

stopwords_english = set(nltk.corpus.stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords_english and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/')
def home():
    return render_template('spamshield.html')

@app.route('/predict_sms', methods=['POST'])
def predict_sms():
    data = request.json
    input_sms = data.get('input_sms')
    if input_sms:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model1.predict(vector_input)[0]
        result_text = "Spam" if result == 1 else "It is safe to proceed"
        return jsonify({'result': result_text})
    return jsonify({'result': 'Invalid input'})

@app.route('/predict_url', methods=['POST'])
def predict_url():
    try:
        data = request.json
        input_url = data.get('input_url')
        if input_url:
            model = joblib.load('model2.pkl')  # Path to your joblib-saved model
            result, percentage = get_prediction(input_url, model)
            result_text = f"Spam - There is {percentage}% chance of being malicious" if result == 'bad' else f"It is safe to proceed"
            return jsonify({'result': result_text})
        return jsonify({'result': 'Invalid input'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
