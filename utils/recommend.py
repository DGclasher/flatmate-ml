import os
import pickle
import numpy as np
import pandas as pd
from app import app
from sklearn.metrics.pairwise import cosine_similarity

app_path = os.path.join(app.root_path, '..', 'datasets', 'user_profiles.csv')
data = pd.read_csv(app_path)

encoder_path = os.path.join(app.root_path, '..', 'dumps', 'encoder.pkl')
scaler_path = os.path.join(app.root_path, '..', 'dumps', 'scaler.pkl')
combined_data_path = os.path.join(
    app.root_path, '..', 'dumps', 'combined_data.pkl')

with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(combined_data_path, "rb") as f:
    X = pickle.load(f)

categorical_columns = [
    'gender', 'occupation', 'educationLevel', 'preferredCity',
    'preferredAccommodation', 'dailySchedule', 'smokingHabits',
    'drinkingHabits', 'dietaryPreferences', 'petPreferences',
    'introvertExtrovert', 'cleanlinessLevel', 'socialHabits',
    'hobbiesAndInterests', 'languagePreferences', 'preferredSecurityMeasures',
    'pastExperiencesWithRoommates'
]
numerical_columns = ['age', 'budgetPriceRange', 'numberOfRoommates']

def recommend_profiles(user_data, num_recommendations=3):
    new_user_data = pd.DataFrame({
        'gender': [user_data['gender']],
        'occupation': [user_data['occupation']],
        'educationLevel': [user_data['educationLevel']],
        'preferredCity': [user_data['preferredCity']],
        'preferredAccommodation': [user_data['preferredAccommodation']],
        'dailySchedule': [user_data['dailySchedule']],
        'smokingHabits': [user_data['smokingHabits']],
        'drinkingHabits': [user_data['drinkingHabits']],
        'dietaryPreferences': [user_data['dietaryPreferences']],
        'petPreferences': [user_data['petPreferences']],
        'introvertExtrovert': [user_data['introvertExtrovert']],
        'cleanlinessLevel': [user_data['cleanlinessLevel']],
        'socialHabits': [user_data['socialHabits']],
        'hobbiesAndInterests': [user_data['hobbiesAndInterests']],
        'languagePreferences': [user_data['languagePreferences']],
        'preferredSecurityMeasures': [user_data['preferredSecurityMeasures']],
        'pastExperiencesWithRoommates': [user_data['pastExperiencesWithRoommates']],
        'age': [user_data['age']],
        'budgetPriceRange': [user_data['budgetPriceRange']],
        'numberOfRoommates': [user_data['numberOfRoommates']]
    })

    new_user_cats = encoder.transform(
        new_user_data[categorical_columns]).toarray()
    new_user_nums = scaler.transform(new_user_data[numerical_columns])
    new_user_X = np.hstack([new_user_cats, new_user_nums])

    new_user_similarities = cosine_similarity(new_user_X, X).flatten()
    similar_user_indices = np.argsort(new_user_similarities)[
        ::-1][:num_recommendations]

    user_ids = data.iloc[similar_user_indices]
    user_ids = user_ids['googleId'].values.tolist()
    return user_ids
