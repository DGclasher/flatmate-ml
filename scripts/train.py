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
    'introvertExtrovert', 'cleanlinessLevel', 'socialHabits', 
    'hobbiesAndInterests', 'languagePreferences', 'preferredSecurityMeasures', 
    'pastExperiencesWithRoommates'
]
numerical_columns = ['age', 'budgetPriceRange', 'numberOfRoommates']

# One-hot encode categorical variables
encoder = OneHotEncoder()
encoded_cats = encoder.fit_transform(data[categorical_columns]).toarray()

# Scale numerical variables
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(data[numerical_columns])

# Combine encoded categorical and scaled numerical data
X = np.hstack([encoded_cats, scaled_nums])

# Compute similarity matrix
similarity_matrix = cosine_similarity(X)

# Function to recommend profiles
def recommend_profiles(user_index, num_recommendations=3):
    user_similarities = similarity_matrix[user_index]
    similar_user_indices = np.argsort(user_similarities)[::-1][1:num_recommendations+1]
    return data.iloc[similar_user_indices]

# Save recommendations for each user in the dataset
recommendations_dict = {}
for i in range(data.shape[0]):
    recommended_profiles = recommend_profiles(i, num_recommendations=3)
    recommendations_dict[data.iloc[i]['googleId']] = recommended_profiles['googleId'].tolist()

# Convert recommendations to a DataFrame
recommendations_df = pd.DataFrame.from_dict(recommendations_dict, orient='index')
recommendations_df.columns = [f'Recommendation_{i+1}' for i in range(recommendations_df.shape[1])]

with open("/app/dumps/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("/app/dumps/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("/app/dumps/combined_data.pkl", "wb") as f:
    pickle.dump(X, f)
    
print("Encoder, scaler, and combined data saved to pickle files.")