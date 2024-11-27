import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

file_path = "/app/datasets/user_profiles.csv"
data = pd.read_csv(file_path)

categorical_columns = [
    'gender', 'occupation', 'educationLevel', 'preferredCity', 
    'preferredAccommodation', 'dailySchedule', 'smokingHabits', 
    'drinkingHabits', 'dietaryPreferences', 'petPreferences', 
    'introvertExtrovert', 'socialHabits', 
    'hobbiesAndInterests', 'languagePreferences', 'preferredSecurityMeasures',
    'isLookingFor',  
]
numerical_columns = ['age', 'budgetPriceRange', 'numberOfRoommates']

encoder = OneHotEncoder()
encoded_cats = encoder.fit_transform(data[categorical_columns]).toarray()

scaler = StandardScaler()
scaled_nums = scaler.fit_transform(data[numerical_columns])

X = np.hstack([encoded_cats, scaled_nums])

similarity_matrix = cosine_similarity(X)

def recommend_profiles(user_index, num_recommendations=3):
    user_is_looking_for = data.iloc[user_index]['isLookingFor']
    complementary_is_looking_for = 'roommate' if user_is_looking_for == 'room' else 'room'
    
    valid_indices = data[data['isLookingFor'] == complementary_is_looking_for].index
    user_similarities = similarity_matrix[user_index, valid_indices]
    
    similar_user_indices = valid_indices[np.argsort(user_similarities)[::-1][:num_recommendations]]
    return data.iloc[similar_user_indices]

recommendations_dict = {}
for i in range(data.shape[0]):
    recommended_profiles = recommend_profiles(i, num_recommendations=3)
    recommendations_dict[data.iloc[i]['googleId']] = recommended_profiles['googleId'].tolist()

recommendations_df = pd.DataFrame.from_dict(recommendations_dict, orient='index')
recommendations_df.columns = [f'Recommendation_{i+1}' for i in range(recommendations_df.shape[1])]

with open("/app/dumps/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("/app/dumps/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("/app/dumps/combined_data.pkl", "wb") as f:
    pickle.dump(X, f)
    
print("Encoder, scaler, and combined data saved to pickle files.")
