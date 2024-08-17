# FlatMate-ML  
![test-package](https://github.com/dgclasher/flatmate-ml/actions/workflows/test-package.yml/badge.svg?branch=main)

FlatMate-ML is the machine learning module of the FlatMate project, responsible for recommending compatible roommates based on user profiles using advanced machine learning techniques.

## Key Features
- **Profile Matching**: Recommends the most compatible roommates by matching user profiles.
- **Data Processing**: Handles data preprocessing, including one-hot encoding for categorical variables and scaling numerical variables.
- **Similarity Calculation**: Computes user profile similarity using cosine similarity.

## Setup & Deployment

### Prerequisites
Ensure you have Docker installed and running on your system.

### Steps to Deploy

1. **Create Required Directories**  
   Set up the necessary directories to store the dataset and dump files:
   ```
   mkdir -p datasets dumps
   ```
2. **Fetch required files**
   Download the sample user_profiles.csv dataset, docker-compose.yml, and .env.example files:
   ```
   wget https://github.com/DGclasher/flatmate-ml/raw/main/datasets/user_profiles.csv -O datasets/ user_profiles.csv
   wget https://github.com/DGclasher/flatmate-ml/raw/main/docker-compose.yml
   wget https://github.com/DGclasher/flatmate-ml/raw/main/.env.example -O .env
   ```
3. **Configure the environment**
   Edit the .env file according to your specific environment settings.

4. **Start the application**
   ```
   docker compose up -d
   ```
5. **To stop the application**
   ```
   docker compose down
   ```