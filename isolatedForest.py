import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib

class TransactionAnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42, 
                 farmer_window_days=7, dealer_window_days=7):
        """
        Initialize the Isolation Forest model with temporal features.
        
        Args:
            contamination: The proportion of outliers in the dataset (default: 0.1 = 10%)
            random_state: Random seed for reproducibility
            farmer_window_days: Rolling window to check farmer transaction frequency
            dealer_window_days: Rolling window to check dealer transaction volume
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
        """
        Create temporal features based on deltaT for both farmers and dealers.
        
        Args:
            df: DataFrame with deltaT column
            
        Returns:
            DataFrame with additional temporal features
        """
        df_temp = df.copy()
        
        # Sort by deltaT to enable rolling calculations
        df_temp = df_temp.sort_values('deltaT').reset_index(drop=True)
        
        # === FARMER TEMPORAL FEATURES ===
        # Count transactions per farmer in the last N days
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
        # Count transactions per dealer in the last N days
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
        # Calculate time between consecutive transactions
        df_temp['farmer_time_since_last'] = df_temp.groupby('farmer_id')['deltaT'].diff()
        df_temp['dealer_time_since_last'] = df_temp.groupby('dealer_id')['deltaT'].diff()
        
        # Fill NaN (first transaction for each entity) with median
        df_temp['farmer_time_since_last'] = df_temp['farmer_time_since_last'].fillna(
            df_temp['farmer_time_since_last'].median()
        )
        df_temp['dealer_time_since_last'] = df_temp['dealer_time_since_last'].fillna(
            df_temp['dealer_time_since_last'].median()
        )
        
        # === RATIO FEATURES ===
        # Average quantity per transaction in window
        df_temp['farmer_avg_qty_window'] = (
            df_temp['farmer_total_qty_window'] / df_temp['farmer_txn_count_window']
        )
        df_temp['dealer_avg_qty_window'] = (
            df_temp['dealer_total_qty_window'] / df_temp['dealer_txn_count_window']
        )
        
        # Dealer concentration: txns per unique farmer
        df_temp['dealer_txn_per_farmer'] = (
            df_temp['dealer_txn_count_window'] / df_temp['dealer_unique_farmers_window']
        )
        
        return df_temp
    
    def preprocess_data(self, df, fit=True):
        """
        Preprocess the data by encoding categorical variables and adding temporal features.
        
        Args:
            df: Input DataFrame
            fit: If True, fit the label encoders. If False, use existing encoders.
        
        Returns:
            Processed DataFrame with encoded features
        """
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
            'deltaT',
            'farmer_txn_count_window',
            'farmer_total_qty_window',
            'farmer_avg_qty_window',
            'farmer_time_since_last',
            'dealer_txn_count_window',
            'dealer_total_qty_window',
            'dealer_avg_qty_window',
            'dealer_unique_farmers_window',
            'dealer_txn_per_farmer',
            'dealer_time_since_last'
        ]
        
        numeric_cols = ['fertilizer_qty_kg']
        
        if fit:
            self.feature_columns = numeric_cols + temporal_cols + encoded_cols
        
        return df_processed[self.feature_columns]
    
    def fit(self, df):
        """
        Train the Isolation Forest model.
        
        Args:
            df: Training DataFrame (must include deltaT column)
        """
        if 'deltaT' not in df.columns:
            raise ValueError("DataFrame must contain 'deltaT' column")
        
        # Preprocess data
        X = self.preprocess_data(df, fit=True)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Train the model
        print(f"Training Isolation Forest on {len(X)} samples with {X.shape[1]} features...")
        print(f"Features include temporal patterns (window: {self.farmer_window_days} days)")
        self.model.fit(X)
        print("Training complete!")
        
        return self
    
    def predict(self, df):
        """
        Predict anomalies in the data.
        
        Args:
            df: DataFrame to predict on
        
        Returns:
            Array of predictions: -1 for anomalies, 1 for normal transactions
        """
        X = self.preprocess_data(df, fit=False)
        X = X.fillna(X.median())
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_with_scores(self, df):
        """
        Predict anomalies and return detailed analysis.
        
        Args:
            df: DataFrame to predict on
        
        Returns:
            DataFrame with predictions, scores, and temporal features
        """
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
        """
        Test different anomaly score thresholds and report statistics.
        
        Args:
            df: DataFrame to analyze
            percentiles: List of percentile thresholds to test
        
        Returns:
            Dictionary with threshold analysis
        """
        results = self.predict_with_scores(df)
        scores = results['anomaly_score'].values
        
        print("\n" + "="*70)
        print("THRESHOLD ANALYSIS - Testing Different Anomaly Cutoffs")
        print("="*70)
        print("Note: Lower anomaly scores = more anomalous")
        print("      Percentile X means bottom (100-X)% are flagged as anomalies")
        
        threshold_results = {}
        
        for percentile in percentiles:
            threshold = np.percentile(scores, percentile)
            # Anomalies have LOWER scores, so we want scores <= threshold
            # But if we use 95th percentile, we want the BOTTOM 5%, not top 5%
            # So we should use (100 - percentile) for the actual cutoff
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
            print(f"  Actual Percentage: {percentage:.2f}%")
        
        return threshold_results
    
    def set_anomaly_threshold(self, df, threshold_score=None, percentile=None):
        """
        Apply a custom threshold to classify anomalies.
        
        Args:
            df: DataFrame to analyze
            threshold_score: Direct anomaly score threshold (lower = more anomalous)
            percentile: Target percentage of anomalies (e.g., 5 = bottom 5% are anomalies)
        
        Returns:
            DataFrame with anomaly classifications based on threshold
        """
        results = self.predict_with_scores(df)
        
        if threshold_score is None and percentile is None:
            raise ValueError("Must provide either threshold_score or percentile")
        
        if percentile is not None:
            # percentile here means "% of data that are anomalies"
            # So percentile=5 means bottom 5% are anomalies
            threshold_score = np.percentile(results['anomaly_score'], percentile)
            print(f"Flagging bottom {percentile}% as anomalies")
            print(f"Threshold score: {threshold_score:.4f}")
        else:
            print(f"Using custom threshold: {threshold_score:.4f}")
        
        # Classify based on threshold (lower scores = more anomalous)
        results['is_anomaly_threshold'] = results['anomaly_score'] <= threshold_score
        results['anomaly'] = results['is_anomaly_threshold'].map({True: -1, False: 1})
        
        anomaly_count = results['is_anomaly_threshold'].sum()
        percentage = (anomaly_count / len(results)) * 100
        
        print(f"\nResults:")
        print(f"  Total Transactions: {len(results):,}")
        print(f"  Anomalies Detected: {anomaly_count:,}")
        print(f"  Percentage: {percentage:.2f}%")
        print(f"  Normal Transactions: {len(results) - anomaly_count:,} ({100-percentage:.2f}%)")
        
        return results
    
    def analyze_anomalies(self, df, threshold_score=None, percentile=None):
        """
        Analyze and categorize detected anomalies with custom threshold.
        
        Args:
            df: DataFrame with predictions
            threshold_score: Custom anomaly score threshold
            percentile: Or percentile threshold (e.g., 95)
        
        Returns:
            Dictionary with anomaly statistics and examples
        """
        if threshold_score is not None or percentile is not None:
            results = self.set_anomaly_threshold(df, threshold_score, percentile)
            anomalies = results[results['is_anomaly_threshold'] == True].copy()
        else:
            results = self.predict_with_scores(df)
            anomalies = results[results['is_anomaly'] == True].copy()
        
        if len(anomalies) == 0:
            return {"message": "No anomalies detected"}
        
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
            anomalies['farmer_time_since_last'] < 1  # Less than 1 day
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
    
    def save_model(self, model_path='isolation_forest_model.pkl', 
                   encoders_path='label_encoders.pkl'):
        """Save the trained model and encoders to disk."""
        joblib.dump(self.model, model_path)
        joblib.dump({
            'encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'farmer_window_days': self.farmer_window_days,
            'dealer_window_days': self.dealer_window_days
        }, encoders_path)
        print(f"Model saved to {model_path}")
        print(f"Encoders saved to {encoders_path}")
    
    def load_model(self, model_path='isolation_forest_model.pkl',
                   encoders_path='label_encoders.pkl'):
        """Load a trained model and encoders from disk."""
        self.model = joblib.load(model_path)
        encoder_data = joblib.load(encoders_path)
        self.label_encoders = encoder_data['encoders']
        self.feature_columns = encoder_data['feature_columns']
        self.categorical_columns = encoder_data['categorical_columns']
        self.farmer_window_days = encoder_data['farmer_window_days']
        self.dealer_window_days = encoder_data['dealer_window_days']
        print("Model and encoders loaded successfully!")


def main():
    """Example usage with threshold analysis"""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('transactions.csv')
    
    print(f"Loaded {len(df)} transactions")
    print(f"Columns: {df.columns.tolist()}")
    
    # Verify deltaT exists
    if 'deltaT' not in df.columns:
        raise ValueError("CSV must contain 'deltaT' column (days since reference date)")
    
    # Initialize detector with 7-day windows
    detector = TransactionAnomalyDetector(
        contamination=0.1,
        farmer_window_days=7,
        dealer_window_days=7
    )
    
    # Train the model
    detector.fit(df)
    
    # STEP 1: Find optimal threshold by testing different percentiles
    print("\n" + "="*70)
    print("STEP 1: FINDING OPTIMAL THRESHOLD")
    print("="*70)
    
    threshold_analysis = detector.find_optimal_threshold(
        df, 
        percentiles=[90, 95, 97, 99, 99.5]  # These will show bottom 10%, 5%, 3%, 1%, 0.5%
    )
    
    # STEP 2: Choose a threshold and get detailed results
    print("\n" + "="*70)
    print("STEP 2: APPLYING CHOSEN THRESHOLD")
    print("="*70)
    
    # Specify what % of data you want flagged as anomalies
    # For example: 12 means "flag the bottom 12% as anomalies"
    ANOMALY_PERCENTAGE = 12  # Change this to whatever % you want
    
    print(f"\nTargeting {ANOMALY_PERCENTAGE}% of transactions as anomalies...")
    analysis, results = detector.analyze_anomalies(df, percentile=ANOMALY_PERCENTAGE)
    
    # STEP 3: Display detailed analysis
    print("\n" + "="*70)
    print("ANOMALY ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total Transactions: {analysis['total_transactions']:,}")
    print(f"  Anomalies Detected: {analysis['total_anomalies']:,}")
    print(f"  Percentage of Anomalies: {analysis['percentage']}")
    print(f"  Normal Transactions: {analysis['total_transactions'] - analysis['total_anomalies']:,}")
    
    print(f"\n🔍 ANOMALY BREAKDOWN:")
    print(f"  Farmer Velocity Anomalies: {analysis['farmer_velocity_anomalies']}")
    print(f"  Dealer Velocity Anomalies: {analysis['dealer_velocity_anomalies']}")
    print(f"  High Quantity Anomalies: {analysis['high_quantity_anomalies']}")
    print(f"  Rapid Repeat Purchase Anomalies: {analysis['rapid_repeat_anomalies']}")
    
    print("\n⚠️  TOP 5 SUSPICIOUS FARMERS (by transaction count):")
    for i, farmer in enumerate(analysis['top_suspicious_farmers'], 1):
        print(f"  {i}. Farmer {farmer['farmer_id']}:")
        print(f"     - Transactions in window: {farmer['farmer_txn_count_window']}")
        print(f"     - Total quantity: {farmer['farmer_total_qty_window']:.0f}kg")
        print(f"     - Anomaly score: {farmer['anomaly_score']:.4f}")
    
    print("\n⚠️  TOP 5 SUSPICIOUS DEALERS (by transaction count):")
    for i, dealer in enumerate(analysis['top_suspicious_dealers'], 1):
        print(f"  {i}. Dealer {dealer['dealer_id']}:")
        print(f"     - Transactions in window: {dealer['dealer_txn_count_window']}")
        print(f"     - Total quantity sold: {dealer['dealer_total_qty_window']:.0f}kg")
        print(f"     - Anomaly score: {dealer['anomaly_score']:.4f}")
    
    # Save everything
    detector.save_model()
    results.to_csv('anomaly_results.csv', index=False)
    
    # Save just the anomalies for easy review
    anomalies_only = results[results['is_anomaly_threshold'] == True]
    anomalies_only.to_csv('anomalies_only.csv', index=False)
    
    print("\n" + "="*70)
    print("✓ Model saved to: isolation_forest_model.pkl")
    print("✓ All results saved to: anomaly_results.csv")
    print("✓ Anomalies only saved to: anomalies_only.csv")
    print("="*70)


if __name__ == "__main__":
    main()