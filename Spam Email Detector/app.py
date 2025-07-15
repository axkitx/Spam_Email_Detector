from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    vec = vectorizer.transform([email])
    proba = model.predict_proba(vec)[0]
    spam_score = proba[1] * 100
    is_spam = spam_score > 50
    prediction_text = f"{spam_score:.2f}% Spam" if is_spam else f"{100 - spam_score:.2f}% Not Spam"
    bar_width = spam_score if is_spam else (100 - spam_score)
    return render_template('index.html', prediction=prediction_text, email=email, bar_width=bar_width, is_spam=is_spam)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

