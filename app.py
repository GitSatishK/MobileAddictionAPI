from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("addiction_model.pkl")

@app.route('/')
def home():
    return "📱 Mobile Addiction Detector API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        screen_time = data['screen_time']
        unlock_count = data['unlock_count']
        night_use = 1 if data['night_use'] == 'yes' else 0
        game_time = data['game_time']
        social_time = data['social_time']

        features = [[screen_time, unlock_count, night_use, game_time, social_time]]
        prediction = model.predict(features)[0]
        
        return jsonify({
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


