import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# -------------------------
# TransactionAnomalyDetector
# -------------------------
class TransactionAnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42,
                 farmer_window_days=7, dealer_window_days=7):
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
        df_temp = df.copy()
        # Ensure deltaT numeric
        df_temp['deltaT'] = pd.to_numeric(df_temp['deltaT'], errors='coerce')
        df_temp = df_temp.sort_values('deltaT').reset_index(drop=True)

        # FARMER features
        farmer_counts = []
        farmer_total_qty = []
        for idx, row in df_temp.iterrows():
            current_time = row['deltaT']
            farmer_id = row['farmer_id']
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

        # DEALER features
        dealer_counts = []
        dealer_total_qty = []
        dealer_unique_farmers = []
        for idx, row in df_temp.iterrows():
            current_time = row['deltaT']
            dealer_id = row['dealer_id']
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

        # Velocity
        df_temp['farmer_time_since_last'] = df_temp.groupby('farmer_id')['deltaT'].diff()
        df_temp['dealer_time_since_last'] = df_temp.groupby('dealer_id')['deltaT'].diff()
        df_temp['farmer_time_since_last'] = df_temp['farmer_time_since_last'].fillna(
            df_temp['farmer_time_since_last'].median()
        )
        df_temp['dealer_time_since_last'] = df_temp['dealer_time_since_last'].fillna(
            df_temp['dealer_time_since_last'].median()
        )

        # Ratio
        df_temp['farmer_avg_qty_window'] = (
            df_temp['farmer_total_qty_window'] / df_temp['farmer_txn_count_window']
        ).replace([np.inf, -np.inf], np.nan)
        df_temp['dealer_avg_qty_window'] = (
            df_temp['dealer_total_qty_window'] / df_temp['dealer_txn_count_window']
        ).replace([np.inf, -np.inf], np.nan)
        df_temp['dealer_txn_per_farmer'] = (
            df_temp['dealer_txn_count_window'] / df_temp['dealer_unique_farmers_window']
        ).replace([np.inf, -np.inf], np.nan)

        return df_temp

    def preprocess_data(self, df, fit=True):
        df_processed = df.copy()
        df_processed = self.engineer_temporal_features(df_processed)

        # Encode categorical columns
        encoded_cols = []
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
                        # handle unseen classes gracefully
                        df_processed[col + '_encoded'] = df_processed[col].astype(str).map(
                            lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1
                        )
                    else:
                        raise ValueError(f"No encoder found for column: {col}")
                encoded_cols.append(col + '_encoded')

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

        # Ensure all feature columns exist in df
        for c in self.feature_columns:
            if c not in df_processed.columns:
                df_processed[c] = np.nan

        return df_processed[self.feature_columns]

    def fit(self, df):
        if 'deltaT' not in df.columns:
            raise ValueError("DataFrame must contain 'deltaT' column")
        X = self.preprocess_data(df, fit=True)
        X = X.fillna(X.median())
        self.model.fit(X)
        return self

    def predict_with_scores(self, df):
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

    def set_anomaly_threshold(self, df, threshold_score=None, percentile=None):
        results = self.predict_with_scores(df)
        if threshold_score is None and percentile is None:
            raise ValueError("Must provide either threshold_score or percentile")
        if percentile is not None:
            threshold_score = np.percentile(results['anomaly_score'], percentile)
        results['is_anomaly_threshold'] = results['anomaly_score'] <= threshold_score
        results['anomaly'] = results['is_anomaly_threshold'].map({True: -1, False: 1})
        return results

    def save_model(self, model_path='isolation_forest_model.pkl', encoders_path='label_encoders.pkl'):
        joblib.dump(self.model, model_path)
        joblib.dump({
            'encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'farmer_window_days': self.farmer_window_days,
            'dealer_window_days': self.dealer_window_days
        }, encoders_path)

    def load_model(self, model_path='isolation_forest_model.pkl', encoders_path='label_encoders.pkl'):
        self.model = joblib.load(model_path)
        encoder_data = joblib.load(encoders_path)
        self.label_encoders = encoder_data['encoders']
        self.feature_columns = encoder_data['feature_columns']
        self.categorical_columns = encoder_data['categorical_columns']
        self.farmer_window_days = encoder_data['farmer_window_days']
        self.dealer_window_days = encoder_data['dealer_window_days']


# -------------------------
# Streamlit UI
# -------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

st.set_page_config(page_title='Fertilizer Transaction Anomaly Detector', layout='wide')
st.title('🛡️ SubsiGuard: Subsidy Leakage Detection')
st.markdown('Upload transactions CSV (must include: farmer_id, dealer_id, deltaT, fertilizer_qty_kg)')

# Sidebar controls
st.sidebar.header('Model & Window Settings')
contamination = st.sidebar.slider('IsolationForest contamination', 0.001, 0.5, 0.1, step=0.001)
farmer_window_days = st.sidebar.number_input('Farmer window (days)', min_value=1, max_value=365, value=7)
dealer_window_days = st.sidebar.number_input('Dealer window (days)', min_value=1, max_value=365, value=7)
random_state = st.sidebar.number_input('Random seed', value=42)

uploaded_file = st.file_uploader('Upload transactions CSV', type=['csv'])

# Example dataset button
if st.button('Load example dataset (synthetic 1000 rows)'):
    # create a simple synthetic dataset for demo
    n = 1000
    rng = np.random.default_rng(42)
    df_demo = pd.DataFrame({
        'farmer_id': rng.integers(1, 201, n).astype(str),
        'dealer_id': rng.integers(1, 51, n).astype(str),
        'deltaT': np.sort(rng.integers(0, 365, n)).astype(float),
        'fertilizer_qty_kg': rng.integers(1, 100, n),
        'crop': rng.choice(['wheat','rice','maize'], n),
        'season': rng.choice(['Kharif','Rabi','Zaid'], n),
        'fertilizer_type': rng.choice(['NPK','Urea','DAP'], n),
        'farmer_village': rng.choice(['V1','V2','V3','V4'], n),
        'dealer_location': rng.choice(['L1','L2','L3'], n)
    })
    st.session_state['df'] = df_demo
    st.success('Loaded example dataset into session')

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success(f'Loaded {len(df)} rows from uploaded CSV')
    except Exception as e:
        st.error(f'Could not read CSV: {e}')

if 'df' not in st.session_state:
    st.info('Upload a CSV or load the example dataset to proceed')
    st.stop()

df = st.session_state['df']
st.subheader('Preview data')
st.dataframe(df.head())

# Basic validation
required_cols = {'farmer_id', 'dealer_id', 'deltaT', 'fertilizer_qty_kg'}
missing = required_cols - set(df.columns)
if missing:
    st.error(f'Missing required columns: {missing}. Please upload a CSV with these columns.')
    st.stop()

# Initialize detector
if 'detector' not in st.session_state:
    st.session_state['detector'] = TransactionAnomalyDetector(
        contamination=contamination,
        random_state=int(random_state),
        farmer_window_days=int(farmer_window_days),
        dealer_window_days=int(dealer_window_days)
    )

detector = st.session_state['detector']

# Train model section
if st.button('Train Isolation Forest'):
    with st.spinner('Engineering features and training model...'):
        try:
            detector.farmer_window_days = int(farmer_window_days)
            detector.dealer_window_days = int(dealer_window_days)
            detector.model = IsolationForest(
                contamination=contamination,
                random_state=int(random_state),
                n_estimators=100
            )
            detector.fit(df)
            st.session_state['trained'] = True
            st.success('Model trained successfully')
        except Exception as e:
            st.error(f'Error during training: {e}')

if not st.session_state.get('trained', False):
    st.warning('Train the model to enable prediction and analysis')

# Threshold & prediction controls
st.sidebar.header('Anomaly Threshold')
use_percentile = st.sidebar.checkbox('Flag bottom X% as anomalies (percentile)', value=True)
if use_percentile:
    anomaly_percent = st.sidebar.slider('Anomaly percentage (%)', min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    anomaly_percent_input = float(anomaly_percent)
else:
    manual_threshold = st.sidebar.number_input('Manual anomaly score threshold (lower = more anomalous)', value=0.0, format='%f')

if st.button('Run analysis'):
    if not st.session_state.get('trained', False):
        st.error('Please train the model first')
    else:
        with st.spinner('Running prediction and thresholding...'):
            try:
                if use_percentile:
                    results = detector.set_anomaly_threshold(df, percentile=anomaly_percent_input)
                else:
                    results = detector.set_anomaly_threshold(df, threshold_score=manual_threshold)

                st.session_state['results'] = results
                st.success('Analysis complete')
            except Exception as e:
                st.error(f'Error during analysis: {e}')

if 'results' in st.session_state:
    results = st.session_state['results']
    st.subheader('Anomaly Results (preview)')
    st.dataframe(results.head(100))

    st.markdown('### Summary statistics')
    total_transactions = len(results)
    detected = int(results['is_anomaly_threshold'].sum())
    pct = detected / total_transactions * 100
    col1, col2, col3 = st.columns(3)
    col1.metric('Total transactions', f'{total_transactions:,}')
    col2.metric('Anomalies detected', f'{detected:,}')
    col3.metric('Percent anomalous', f'{pct:.2f}%')

    st.markdown('### Top suspicious farmers (by txn count in window)')
    try:
        top_farmers = results[results['is_anomaly_threshold']].nlargest(10, 'farmer_txn_count_window')
        st.dataframe(top_farmers[['farmer_id','farmer_txn_count_window','farmer_total_qty_window','anomaly_score']].head(10))
    except Exception:
        st.info('No anomalies or feature missing')

    st.markdown('### Top suspicious dealers (by txn count in window)')
    try:
        top_dealers = results[results['is_anomaly_threshold']].nlargest(10, 'dealer_txn_count_window')
        st.dataframe(top_dealers[['dealer_id','dealer_txn_count_window','dealer_total_qty_window','anomaly_score']].head(10))
    except Exception:
        st.info('No anomalies or feature missing')

    # Download buttons
    csv_bytes = df_to_csv_bytes(results)
    st.download_button('Download full results (CSV)', data=csv_bytes, file_name='anomaly_results.csv')

    anomalies_only = results[results['is_anomaly_threshold']]
    if not anomalies_only.empty:
        st.download_button('Download anomalies only (CSV)', data=df_to_csv_bytes(anomalies_only), file_name='anomalies_only.csv')

    # Save model
    if st.button('Save trained model to disk'):
        detector.save_model()
        st.success('Model and encoders saved (isolation_forest_model.pkl, label_encoders.pkl)')

st.markdown('---')
st.caption('How to run: 1) Install requirements: pip install streamlit scikit-learn pandas joblib 2) streamlit run app.py')

st.subheader("Distribution of Fertilizer Quantity per Transaction")
fig, ax = plt.subplots()
ax.hist(df['fertilizer_qty_kg'], bins=40, color='skyblue', edgecolor='black')
ax.set_xlabel("Quantity (KG)")
ax.set_ylabel("Number of Transactions")
ax.set_title("Histogram of Fertilizer Quantity Sold")
st.pyplot(fig)

# Dealer Chart
dealer_summary = df.groupby('dealer_id').agg(
    Avg_KG_per_Tx=('fertilizer_qty_kg', 'mean'),
    Total_Transactions=('dealer_id', 'count')
).reset_index()

avg = dealer_summary.sort_values("Avg_KG_per_Tx", ascending=False).head(20)

st.subheader("Top 20 Dealers by Average KG per Transaction")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(avg["dealer_id"], avg["Avg_KG_per_Tx"])
ax.set_xticklabels(avg["dealer_id"], rotation=45, ha='right')
ax.set_ylabel("Avg KG per Transaction")
ax.set_title("Top Dealers by Avg KG/Tx")
st.pyplot(fig)

# ==========================================
# 🟢 NEW: GEOSPATIAL MAP SECTION
# ==========================================
st.markdown("---")
st.subheader("📍 Geospatial Fraud Hotspots")
st.write("Visualizing high-risk villages on the map.")

if 'results' in st.session_state:
    results = st.session_state['results']
    # Filter only anomalies for the map
    anomalies_map = results[results['is_anomaly_threshold'] == True].copy()
    
    if not anomalies_map.empty:
        # 1. Try to load real coordinates
        try:
            geo_df = pd.read_csv('village_coordinates.csv')
            # Merge
            mapped_df = pd.merge(anomalies_map, geo_df, left_on='farmer_village', right_on='Village', how='left')
            mapped_df = mapped_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
        except:
            # 2. Fallback Logic: Maps V1..V4 OR specific names to India coordinates
            fallback_coords = {
                # Synthetic/Demo codes
                'V1': [12.9716, 77.5946], # Bangalore
                'V2': [13.0827, 80.2707], # Chennai
                'V3': [19.0760, 72.8777], # Mumbai
                'V4': [28.7041, 77.1025], # Delhi
                
                # Names from data_gen.py
                'Lakshmipura': [13.0879, 77.5312],
                'Rampur': [13.0562, 77.6887],
                'Bikaner': [28.0241, 73.3123],
                'Khandwa': [21.8316, 76.3493],
                'Mandya': [12.5214, 76.8966],
                'Dharwad': [15.4584, 75.0072]
            }
            
            # Default to India Center if name not found (Avoids Ocean/0,0)
            INDIA_CENTER = [20.5937, 78.9629]

            # Create lat/lon columns manually for fallback
            anomalies_map['latitude'] = anomalies_map['farmer_village'].map(lambda x: fallback_coords.get(x, INDIA_CENTER)[0])
            anomalies_map['longitude'] = anomalies_map['farmer_village'].map(lambda x: fallback_coords.get(x, INDIA_CENTER)[1])
            mapped_df = anomalies_map

        # Drop rows where we couldn't match coordinates or ended up at 0,0
        mapped_df = mapped_df[(mapped_df['latitude'] != 0) & (mapped_df['longitude'] != 0)]
        mapped_df = mapped_df.dropna(subset=['latitude', 'longitude'])
        
        # 3. Show Map
        if not mapped_df.empty:
            st.map(mapped_df)
            st.write(f"Showing {len(mapped_df)} high-risk transactions on map.")
        else:
            st.warning("Could not match village names to coordinates. Ensure 'village_coordinates.csv' exists or use the Example Dataset.")
    else:
        st.info("No anomalies detected yet to place on the map.")
else:
    st.info("Run the analysis above to generate map data.")