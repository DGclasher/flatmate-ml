import os
import pickle
import numpy as np
import pandas as pd
from app import app
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app_path = os.path.join(app.root_path, '..', 'datasets', 'user_profiles.csv')
data = pd.read_csv(app_path)

encoder_path = os.path.join(app.root_path, '..', 'dumps', 'encoder.pkl')
scaler_path = os.path.join(app.root_path, '..', 'dumps', 'scaler.pkl')
combined_data_path = os.path.join(app.root_path, '..', 'dumps', 'combined_data.pkl')

with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(combined_data_path, "rb") as f:
    X = pickle.load(f)

categorical_columns = [
    'Gender', 'Occupation', 'Education_Level', 'Preferred_City',
    'Proximity_to_Workplace/College', 'Preferred_Accommodation',
    'Daily_Schedule', 'Smoking_Habits', 'Drinking_Habits',
    'Dietary_Preferences', 'Pet_Preferences', 'Introvert/Extrovert',
    'Cleanliness_Level', 'Social_Habits', 'Hobbies_and_Interests',
    'Language_Preferences', 'Preferred_Security_Measures',
    'Past_Experiences_with_Roommates'
]
numerical_columns = ['Age', 'Budget/Price_Range', 'Number_of_Roommates']

def recommend_profiles(user_data, num_recommendations=3):
    new_user_data = pd.DataFrame({
        'Gender': [user_data['Gender']],
        'Occupation': [user_data['Occupation']],
        'Education_Level': [user_data['Education_Level']],
        'Preferred_City': [user_data['Preferred_City']],
        'Proximity_to_Workplace/College': [user_data['Proximity_to_Workplace/College']],
        'Preferred_Accommodation': [user_data['Preferred_Accommodation']],
        'Daily_Schedule': [user_data['Daily_Schedule']],
        'Smoking_Habits': [user_data['Smoking_Habits']],
        'Drinking_Habits': [user_data['Drinking_Habits']],
        'Dietary_Preferences': [user_data['Dietary_Preferences']],
        'Pet_Preferences': [user_data['Pet_Preferences']],
        'Introvert/Extrovert': [user_data['Introvert/Extrovert']],
        'Cleanliness_Level': [user_data['Cleanliness_Level']],
        'Social_Habits': [user_data['Social_Habits']],
        'Hobbies_and_Interests': [user_data['Hobbies_and_Interests']],
        'Language_Preferences': [user_data['Language_Preferences']],
        'Preferred_Security_Measures': [user_data['Preferred_Security_Measures']],
        'Past_Experiences_with_Roommates': [user_data['Past_Experiences_with_Roommates']],
        'Age': [user_data['Age']],
        'Budget/Price_Range': [user_data['Budget/Price_Range']],
        'Number_of_Roommates': [user_data['Number_of_Roommates']]
    })

    new_user_cats = encoder.transform(new_user_data[categorical_columns]).toarray()
    new_user_nums = scaler.transform(new_user_data[numerical_columns])
    new_user_X = np.hstack([new_user_cats, new_user_nums])

    new_user_similarities = cosine_similarity(new_user_X, X).flatten()
    similar_user_indices = np.argsort(new_user_similarities)[::-1][:num_recommendations]

    user_ids = data.iloc[similar_user_indices]
    user_ids = user_ids['User_ID'].values
    user_ids = user_ids.tolist()
    return user_ids
