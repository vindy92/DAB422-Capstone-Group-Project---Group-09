from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model, encoder, and power transformer
rf_model = joblib.load('best_rf_model.pkl')
encoder = joblib.load('target_encoder.pkl')
pt = joblib.load('power_transformer.pkl')

# Function to preprocess input data
def preprocess_input(data):
    # Convert input to DataFrame
    input_df = pd.DataFrame(data, index=[0])

    # Encode categorical features
    input_df = encoder.transform(input_df)

    # Feature engineering
    input_df['dance_energy'] = input_df['danceability'] * input_df['energy']
    input_df['valence_tempo'] = input_df['valence'] * input_df['tempo']
    input_df['speechiness_acoustic'] = input_df['speechiness'] * input_df['acousticness']

    # Normalize skewed features
    skewed_features = ['tempo', 'speechiness', 'energy', 'liveness', 'duration_ms']
    input_df[skewed_features] = pt.transform(input_df[skewed_features])

    return input_df

# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for prediction form
@app.route('/home', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Convert minutes and seconds to milliseconds
        try:
            minutes = int(request.form['duration_minutes'])
            seconds = int(request.form['duration_seconds'])
        except ValueError:
            return render_template('index.html', prediction="Invalid input for duration.")

        total_duration_ms = (minutes * 60 + seconds) * 1000

        # Validate duration
        if total_duration_ms < 37702 or total_duration_ms > 655440:
            return render_template('index.html', prediction="Please enter a duration between 0.63 and 10.92 minutes.")

        # Get other form inputs
        data = {
            'artist_name': request.form['artist_name'],
            'country_name': request.form['country_name'],
            'artist_genres': request.form['artist_genres'],
            'speechiness': float(request.form['speechiness']),
            'energy': float(request.form['energy']),
            'valence': float(request.form['valence']),
            'danceability': float(request.form['danceability']),
            'acousticness': float(request.form['acousticness']),
            'tempo': float(request.form['tempo']),
            'liveness': float(request.form['liveness']),
            'duration_ms': total_duration_ms
        }

        # Preprocess input data
        processed_data = preprocess_input(data)

        # Predict using model
        prediction = rf_model.predict(processed_data)
        clamped_prediction = max(0, min(100, prediction[0]))
        formatted_prediction = f"{clamped_prediction:.2f}%"

        return render_template('index.html', prediction=formatted_prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
