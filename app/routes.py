from app import app
from flask import jsonify, request

from utils.recommend import recommend_profiles

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    users = recommend_profiles(data)
    return jsonify({'users': users}), 200