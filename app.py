import os
import secrets
from flask import Flask, jsonify, request
from utils.recommend import load_resources, recommend_profiles

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['ROOT_DIR'] = os.path.dirname(os.path.abspath(__file__))

data, encoder, scaler, X = load_resources(app.config['ROOT_DIR'])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/recommend', methods=['POST'])
def recommend():
    user_data = request.get_json()
    users = recommend_profiles(user_data, data, encoder, scaler, X)
    return jsonify({'users': users}), 200


if __name__ == '__main__':
    app.run()
