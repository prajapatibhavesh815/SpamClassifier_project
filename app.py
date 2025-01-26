from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string
import os

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to transform text
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
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Function to predict spam
def predict_spam(message):
    transformed_sms = transform_text(message)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    return result

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        return render_template('index.html', result=result)

if __name__ == '__main__':
    # Define paths dynamically using os.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tfidf_path = os.path.join(current_dir, 'vectorizer.pkl')
    model_path = os.path.join(current_dir, 'model.pkl')

    # Load the model and vectorizer
    tfidf = pickle.load(open(tfidf_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))

    # For production, use Waitress (comment the app.run line)
    from waitress import serve
    serve(app, host='0.0.0.0', port=8000)
