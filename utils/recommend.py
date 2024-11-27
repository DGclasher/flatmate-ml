import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_resources(root_dir):
    app_path = os.path.join(root_dir, 'datasets', 'user_profiles.csv')
    data = pd.read_csv(app_path)

    encoder_path = os.path.join(root_dir, 'dumps', 'encoder.pkl')
    scaler_path = os.path.join(root_dir, 'dumps', 'scaler.pkl')
    combined_data_path = os.path.join(root_dir, 'dumps', 'combined_data.pkl')

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(combined_data_path, "rb") as f:
        X = pickle.load(f)

    return data, encoder, scaler, X

def recommend_profiles(user_data, data, encoder, scaler, X, num_recommendations=3):
    categorical_columns = [
        'gender', 'occupation', 'educationLevel', 'preferredCity',
        'preferredAccommodation', 'dailySchedule', 'smokingHabits',
        'drinkingHabits', 'dietaryPreferences', 'petPreferences',
        'introvertExtrovert', 'socialHabits',
        'hobbiesAndInterests', 'languagePreferences', 'preferredSecurityMeasures',
        'isLookingFor'
    ]
    numerical_columns = ['age', 'budgetPriceRange', 'numberOfRoommates']

    complementary_is_looking_for = 'roommate' if user_data['isLookingFor'] == 'room' else 'room'
    valid_indices = data[data['isLookingFor'] == complementary_is_looking_for].index

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
        'socialHabits': [user_data['socialHabits']],
        'hobbiesAndInterests': [user_data['hobbiesAndInterests']],
        'languagePreferences': [user_data['languagePreferences']],
        'preferredSecurityMeasures': [user_data['preferredSecurityMeasures']],
        'isLookingFor': [user_data['isLookingFor']],
        'age': [user_data['age']],
        'budgetPriceRange': [user_data['budgetPriceRange']],
        'numberOfRoommates': [user_data['numberOfRoommates']]
    })

    new_user_cats = encoder.transform(new_user_data[categorical_columns]).toarray()
    new_user_nums = scaler.transform(new_user_data[numerical_columns])
    new_user_X = np.hstack([new_user_cats, new_user_nums])

    valid_X = X[valid_indices]
    new_user_similarities = cosine_similarity(new_user_X, valid_X).flatten()
    similar_user_indices = valid_indices[np.argsort(new_user_similarities)[::-1][:num_recommendations]]

    user_ids = data.iloc[similar_user_indices]['googleId'].values.tolist()
    return user_ids
