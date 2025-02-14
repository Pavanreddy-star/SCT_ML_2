import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    # Selecting relevant features
    df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Scaling features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    return scaled_features, df

if __name__ == "__main__":
    X, df = load_and_preprocess("data/customer_data.csv")  # Corrected path
    print("Data Preprocessing Completed")
