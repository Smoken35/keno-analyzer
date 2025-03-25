import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keno_data/processing.log'),
        logging.StreamHandler()
    ]
)

class KenoDataProcessor:
    def __init__(self):
        self.data_dir = Path("keno_data")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
        
    def load_data(self, file_pattern="keno_data_*.csv"):
        """Load the most recent Keno data file."""
        data_files = list(self.data_dir.glob(file_pattern))
        if not data_files:
            raise FileNotFoundError("No Keno data files found")
        
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        logging.info(f"Loading data from {latest_file}")
        return pd.read_csv(latest_file)
    
    def extract_features(self, df):
        """Extract features from the raw data."""
        # Convert date and time to datetime
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        # Extract time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        # Create features from the numbers
        numbers_df = pd.DataFrame(df['numbers'].tolist(), columns=[f'number_{i+1}' for i in range(20)])
        
        # Calculate statistical features
        df['mean'] = numbers_df.mean(axis=1)
        df['std'] = numbers_df.std(axis=1)
        df['min'] = numbers_df.min(axis=1)
        df['max'] = numbers_df.max(axis=1)
        
        # Calculate frequency of each number
        for i in range(1, 81):
            df[f'freq_{i}'] = numbers_df.apply(lambda x: (x == i).sum(), axis=1)
        
        # Combine all features
        feature_columns = ['hour', 'day_of_week', 'month', 'mean', 'std', 'min', 'max'] + \
                         [f'freq_{i}' for i in range(1, 81)]
        
        return df[feature_columns]
    
    def prepare_training_data(self, df, target_size=20):
        """Prepare data for training the model."""
        X = self.extract_features(df)
        
        # Create target variables (next draw's numbers)
        y = []
        for i in range(len(df) - 1):
            y.append(df['numbers'].iloc[i + 1])
        y = np.array(y)
        
        # Remove the last row from X since we don't have target for it
        X = X.iloc[:-1]
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Save the scaler for later use
        joblib.dump(self.scaler, self.models_dir / 'scaler.joblib')
        
        return X_scaled, y
    
    def process_data(self):
        """Main method to process the data."""
        try:
            # Load raw data
            df = self.load_data()
            
            # Prepare training data
            X, y = self.prepare_training_data(df)
            
            # Save processed data
            np.save(self.models_dir / 'X.npy', X)
            np.save(self.models_dir / 'y.npy', y)
            
            logging.info(f"Processed data shape: X={X.shape}, y={y.shape}")
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            raise

def main():
    processor = KenoDataProcessor()
    X, y = processor.process_data()
    logging.info("Data processing completed successfully")

if __name__ == "__main__":
    main() 