import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib

# ==========================================
# 1. THE BRAIN: Fraud Detection Class
# ==========================================
class TransactionAnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42, 
                 farmer_window_days=7, dealer_window_days=7):
        """
        Initialize the Isolation Forest model with temporal features.
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.label_encoders = {}
        self.categorical_columns = ['crop', 'season', 'fertilizer_type', 
                                   'farmer_village', 'dealer_location']
        self.feature_columns = None
        self.farmer_window_days = farmer_window_days
        self.dealer_window_days = dealer_window_days
        
    def engineer_temporal_features(self, df):
        """Create temporal features based on deltaT for both farmers and dealers."""
        df_temp = df.copy()
        
        # Sort by deltaT to enable rolling calculations
        df_temp = df_temp.sort_values('deltaT').reset_index(drop=True)
        
        # === FARMER TEMPORAL FEATURES ===
        farmer_counts = []
        farmer_total_qty = []
        
        for idx, row in df_temp.iterrows():
            current_time = row['deltaT']
            farmer_id = row['farmer_id']
            
            # Get farmer's transactions in the window
            farmer_mask = (
                (df_temp['farmer_id'] == farmer_id) & 
                (df_temp['deltaT'] >= current_time - self.farmer_window_days) &
                (df_temp['deltaT'] <= current_time)
            )
            
            farmer_window = df_temp[farmer_mask]
            farmer_counts.append(len(farmer_window))
            farmer_total_qty.append(farmer_window['fertilizer_qty_kg'].sum())
        
        df_temp['farmer_txn_count_window'] = farmer_counts
        df_temp['farmer_total_qty_window'] = farmer_total_qty
        
        # === DEALER TEMPORAL FEATURES ===
        dealer_counts = []
        dealer_total_qty = []
        dealer_unique_farmers = []
        
        for idx, row in df_temp.iterrows():
            current_time = row['deltaT']
            dealer_id = row['dealer_id']
            
            # Get dealer's transactions in the window
            dealer_mask = (
                (df_temp['dealer_id'] == dealer_id) & 
                (df_temp['deltaT'] >= current_time - self.dealer_window_days) &
                (df_temp['deltaT'] <= current_time)
            )
            
            dealer_window = df_temp[dealer_mask]
            dealer_counts.append(len(dealer_window))
            dealer_total_qty.append(dealer_window['fertilizer_qty_kg'].sum())
            dealer_unique_farmers.append(dealer_window['farmer_id'].nunique())
        
        df_temp['dealer_txn_count_window'] = dealer_counts
        df_temp['dealer_total_qty_window'] = dealer_total_qty
        df_temp['dealer_unique_farmers_window'] = dealer_unique_farmers
        
        # === VELOCITY FEATURES ===
        df_temp['farmer_time_since_last'] = df_temp.groupby('farmer_id')['deltaT'].diff()
        df_temp['dealer_time_since_last'] = df_temp.groupby('dealer_id')['deltaT'].diff()
        
        # Fill NaN (first transaction) with median
        df_temp['farmer_time_since_last'] = df_temp['farmer_time_since_last'].fillna(
            df_temp['farmer_time_since_last'].median()
        )
        df_temp['dealer_time_since_last'] = df_temp['dealer_time_since_last'].fillna(
            df_temp['dealer_time_since_last'].median()
        )
        
        # === RATIO FEATURES ===
        df_temp['farmer_avg_qty_window'] = (
            df_temp['farmer_total_qty_window'] / df_temp['farmer_txn_count_window']
        )
        df_temp['dealer_avg_qty_window'] = (
            df_temp['dealer_total_qty_window'] / df_temp['dealer_txn_count_window']
        )
        df_temp['dealer_txn_per_farmer'] = (
            df_temp['dealer_txn_count_window'] / df_temp['dealer_unique_farmers_window']
        )
        
        return df_temp
    
    def preprocess_data(self, df, fit=True):
        """Preprocess the data by encoding categorical variables and adding temporal features."""
        df_processed = df.copy()
        
        # Add temporal features
        df_processed = self.engineer_temporal_features(df_processed)
        
        # Encode categorical columns
        for col in self.categorical_columns:
            if col in df_processed.columns:
                if fit:
                    le = LabelEncoder()
                    df_processed[col + '_encoded'] = le.fit_transform(
                        df_processed[col].astype(str)
                    )
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df_processed[col + '_encoded'] = df_processed[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        raise ValueError(f"No encoder found for column: {col}")
        
        # Select feature columns
        encoded_cols = [col + '_encoded' for col in self.categorical_columns 
                        if col in df.columns]
        
        temporal_cols = [
            'deltaT', 'farmer_txn_count_window', 'farmer_total_qty_window',
            'farmer_avg_qty_window', 'farmer_time_since_last',
            'dealer_txn_count_window', 'dealer_total_qty_window',
            'dealer_avg_qty_window', 'dealer_unique_farmers_window',
            'dealer_txn_per_farmer', 'dealer_time_since_last'
        ]
        
        numeric_cols = ['fertilizer_qty_kg']
        
        if fit:
            self.feature_columns = numeric_cols + temporal_cols + encoded_cols
        
        return df_processed[self.feature_columns]
    
    def fit(self, df):
        """Train the Isolation Forest model."""
        if 'deltaT' not in df.columns:
            raise ValueError("DataFrame must contain 'deltaT' column")
        
        # Preprocess data
        X = self.preprocess_data(df, fit=True)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Train the model
        print(f"Training Isolation Forest on {len(X)} samples with {X.shape[1]} features...")
        self.model.fit(X)
        print("Training complete!")
        return self
    
    def predict(self, df):
        X = self.preprocess_data(df, fit=False)
        X = X.fillna(X.median())
        predictions = self.model.predict(X)
        return predictions
    
    def predict_with_scores(self, df):
        # Get processed features for analysis
        df_with_features = self.engineer_temporal_features(df.copy())
        
        X = self.preprocess_data(df, fit=False)
        X = X.fillna(X.median())
        
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        result_df = df_with_features.copy()
        result_df['anomaly'] = predictions
        result_df['anomaly_score'] = scores
        result_df['is_anomaly'] = predictions == -1
        
        return result_df
    
    def find_optimal_threshold(self, df, percentiles=[90, 95, 97, 99]):
        results = self.predict_with_scores(df)
        scores = results['anomaly_score'].values
        
        print("\n" + "="*70)
        print("THRESHOLD ANALYSIS - Testing Different Anomaly Cutoffs")
        print("="*70)
        
        threshold_results = {}
        for percentile in percentiles:
            threshold = np.percentile(scores, percentile)
            actual_percentile = 100 - percentile
            actual_threshold = np.percentile(scores, actual_percentile)
            
            anomalies = results[results['anomaly_score'] <= actual_threshold]
            percentage = (len(anomalies) / len(results)) * 100
            
            threshold_results[percentile] = {
                'threshold_value': actual_threshold,
                'count': len(anomalies),
                'percentage': percentage
            }
            print(f"\nTarget: Bottom {100-percentile}% as anomalies")
            print(f"  Threshold Score: {actual_threshold:.4f}")
            print(f"  Anomalies Detected: {len(anomalies):,} / {len(results):,}")
        
        return threshold_results
    
    def set_anomaly_threshold(self, df, threshold_score=None, percentile=None):
        results = self.predict_with_scores(df)
        
        if threshold_score is None and percentile is None:
            raise ValueError("Must provide either threshold_score or percentile")
        
        if percentile is not None:
            threshold_score = np.percentile(results['anomaly_score'], percentile)
            print(f"Flagging bottom {percentile}% as anomalies (Score < {threshold_score:.4f})")
        else:
            print(f"Using custom threshold: {threshold_score:.4f}")
        
        results['is_anomaly_threshold'] = results['anomaly_score'] <= threshold_score
        results['anomaly'] = results['is_anomaly_threshold'].map({True: -1, False: 1})
        
        return results
    
    def analyze_anomalies(self, df, threshold_score=None, percentile=None):
        if threshold_score is not None or percentile is not None:
            results = self.set_anomaly_threshold(df, threshold_score, percentile)
            anomalies = results[results['is_anomaly_threshold'] == True].copy()
        else:
            results = self.predict_with_scores(df)
            anomalies = results[results['is_anomaly'] == True].copy()
        
        if len(anomalies) == 0:
            return {"message": "No anomalies detected"}, results
        
        # Categorize anomalies
        farmer_velocity = anomalies[
            anomalies['farmer_txn_count_window'] > anomalies['farmer_txn_count_window'].quantile(0.9)
        ]
        dealer_velocity = anomalies[
            anomalies['dealer_txn_count_window'] > anomalies['dealer_txn_count_window'].quantile(0.9)
        ]
        high_quantity = anomalies[
            anomalies['fertilizer_qty_kg'] > anomalies['fertilizer_qty_kg'].quantile(0.95)
        ]
        rapid_repeat = anomalies[
            anomalies['farmer_time_since_last'] < 1 
        ]
        
        analysis = {
            "total_transactions": len(results),
            "total_anomalies": len(anomalies),
            "percentage": f"{len(anomalies)/len(results)*100:.2f}%",
            "farmer_velocity_anomalies": len(farmer_velocity),
            "dealer_velocity_anomalies": len(dealer_velocity),
            "high_quantity_anomalies": len(high_quantity),
            "rapid_repeat_anomalies": len(rapid_repeat),
            "top_suspicious_farmers": anomalies.nlargest(5, 'farmer_txn_count_window')[
                ['farmer_id', 'farmer_txn_count_window', 'farmer_total_qty_window', 'anomaly_score']
            ].to_dict('records'),
            "top_suspicious_dealers": anomalies.nlargest(5, 'dealer_txn_count_window')[
                ['dealer_id', 'dealer_txn_count_window', 'dealer_total_qty_window', 'anomaly_score']
            ].to_dict('records')
        }
        return analysis, results
    
    def save_model(self, model_path='isolation_forest_model.pkl', encoders_path='label_encoders.pkl'):
        joblib.dump(self.model, model_path)
        joblib.dump({
            'encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'farmer_window_days': self.farmer_window_days,
            'dealer_window_days': self.dealer_window_days
        }, encoders_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='isolation_forest_model.pkl', encoders_path='label_encoders.pkl'):
        self.model = joblib.load(model_path)
        encoder_data = joblib.load(encoders_path)
        self.label_encoders = encoder_data['encoders']
        self.feature_columns = encoder_data['feature_columns']
        self.categorical_columns = encoder_data['categorical_columns']
        self.farmer_window_days = encoder_data['farmer_window_days']
        self.dealer_window_days = encoder_data['dealer_window_days']
        print("Model and encoders loaded successfully!")

# ==========================================
# 2. THE MAP HELPER FUNCTION
# ==========================================
def add_geolocation(suspicious_transactions_df):
    """
    Takes the list of fraud transactions and adds GPS coordinates.
    """
    print("\n🗺️  Mapping coordinates to anomalies...")
    
    # 1. Load the Village Coordinates CSV
    try:
        # Check if the file exists
        geo_df = pd.read_csv('data/village_coordinates.csv') 
    except FileNotFoundError:
        # Try local folder if data folder fails
        try:
             geo_df = pd.read_csv('village_coordinates.csv')
        except FileNotFoundError:
            print("⚠️ Warning: 'village_coordinates.csv' not found. Using dummy data.")
            data = {
                'Village': ['Rampur', 'Shyamgarh'], 
                'Latitude': [18.52, 18.53], 
                'Longitude': [73.85, 73.84]
            }
            geo_df = pd.DataFrame(data)

    # 2. Merge fraud data with location data
    # IMPORTANT: Ensure your transactions.csv has 'farmer_village' 
    # and village_coordinates.csv has 'Village'
    mapped_df = pd.merge(
        suspicious_transactions_df, 
        geo_df, 
        left_on='farmer_village', 
        right_on='Village', 
        how='left'
    )
    
    # 3. Rename columns for Streamlit (needs lowercase 'latitude', 'longitude')
    mapped_df = mapped_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    
    # 4. Remove rows without coordinates
    original_len = len(mapped_df)
    mapped_df = mapped_df.dropna(subset=['latitude', 'longitude'])
    print(f"   Matches found: {len(mapped_df)} / {original_len} locations")
    
    return mapped_df

# ==========================================
# 3. THE MAIN EXECUTION BLOCK
# ==========================================
def main():
    """Main execution pipeline"""
    
    # --- Step 1: Load Data ---
    print("Loading data...")
    try:
        df = pd.read_csv('transactions.csv')
    except FileNotFoundError:
        print("❌ Error: transactions.csv not found!")
        return

    print(f"Loaded {len(df)} transactions")
    
    # Verify deltaT exists
    if 'deltaT' not in df.columns:
        raise ValueError("CSV must contain 'deltaT' column (days since reference date)")
    
    # --- Step 2: Train Detector ---
    detector = TransactionAnomalyDetector(
        contamination=0.1,
        farmer_window_days=7,
        dealer_window_days=7
    )
    
    detector.fit(df)
    
    # --- Step 3: Find Threshold ---
    print("\n" + "="*70)
    print("STEP 1: FINDING OPTIMAL THRESHOLD")
    print("="*70)
    
    detector.find_optimal_threshold(df, percentiles=[90, 95, 97, 99])
    
    # --- Step 4: Apply Logic ---
    print("\n" + "="*70)
    print("STEP 2: APPLYING CHOSEN THRESHOLD")
    print("="*70)
    
    # We flag the bottom 5% as anomalies (adjust this number if needed)
    ANOMALY_PERCENTAGE = 5 
    
    print(f"\nTargeting {ANOMALY_PERCENTAGE}% of transactions as anomalies...")
    analysis, results = detector.analyze_anomalies(df, percentile=ANOMALY_PERCENTAGE)
    
    # --- Step 5: Show Results ---
    print("\n" + "="*70)
    print("ANOMALY ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total Transactions: {analysis['total_transactions']:,}")
    print(f"  Anomalies Detected: {analysis['total_anomalies']:,}")
    print(f"  Percentage: {analysis['percentage']}")
    
    print("\n⚠️  TOP 5 SUSPICIOUS FARMERS:")
    for i, farmer in enumerate(analysis['top_suspicious_farmers'], 1):
        print(f"  {i}. Farmer {farmer['farmer_id']} (Score: {farmer['anomaly_score']:.4f})")
    
    # --- Step 6: Save Results ---
    detector.save_model()
    results.to_csv('anomaly_results.csv', index=False)
    
    # Save just the anomalies for the dashboard list
    anomalies_only = results[results['is_anomaly_threshold'] == True]
    anomalies_only.to_csv('anomalies_only.csv', index=False)
    
    # --- Step 7: RUN THE MAP LOGIC ---
    # This creates the file for the map visualization
    mapped_anomalies = add_geolocation(anomalies_only)
    mapped_anomalies.to_csv('mapped_anomalies.csv', index=False)
    print("✓ Map data saved to: mapped_anomalies.csv")
    
    print("\n" + "="*70)
    print("✓ SUCCESS: All 3 output files generated!")
    print("="*70)

if __name__ == "__main__":
    main()