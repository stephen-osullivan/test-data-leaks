# simple endpoint app
from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

def load_model(f = 'model.joblib'):
    model = load(f)
    return model

@app.route('/')
def home():
    """
        this is the home page
    """
    return "Hello MLops 01!"

@app.route('/predict', methods = ['POST'])
def predict():
    data = pd.DataFrame.from_dict(request.get_json())
    model = load_model()
    pred = model.predict(data)
    return jsonify({'predictions':list(pred)})

def send_info(name, email, phone):
    # Example usage
    #name = "John Doe"
    #email = "john.doe@example.com"
    #phone = "555-1234"
    #send_personal_info(name, email, phone)
    # Simulate sending personal information via email
    print(f"Sending information to {email}...")
    
    # Dummy implementation - just printing the information
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Phone: {phone}")


# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
