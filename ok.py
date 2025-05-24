# app.py

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Data
races = pd.read_csv("csv/races.csv")
results = pd.read_csv("csv/results.csv")
status = pd.read_csv("csv/status.csv")
drivers = pd.read_csv("csv/drivers.csv")
circuits = pd.read_csv("csv/circuits.csv")

# Preprocessing
circuits.rename(columns={'name': 'circuit_name'}, inplace=True)

# Merge datasets
merged_df = results.merge(
    races[['raceId', 'year', 'round', 'name', 'circuitId', 'date']],
    on='raceId', how='left'
)
merged_df = merged_df.merge(status, on='statusId', how='left')
merged_df.drop(columns=[col for col in [
    'resultId', 'number', 'position', 'positionText', 'time', 'milliseconds',
    'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed', 'statusId', 'url'
] if col in merged_df.columns], inplace=True)

merged_df['winner'] = (merged_df['positionOrder'] == 1).astype(int)
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
merged_df.sort_values(by=['year', 'round'], inplace=True)

# Feature Engineering
df = merged_df.copy()
df['driver_total_races'] = df.groupby('driverId').cumcount() + 1
df['driver_total_wins'] = df.groupby('driverId')['winner'].cumsum()
df['driver_win_rate'] = df['driver_total_wins'] / df['driver_total_races']
df['constructor_total_races'] = df.groupby('constructorId').cumcount() + 1
df['constructor_total_wins'] = df.groupby('constructorId')['winner'].cumsum()
df['constructor_win_rate'] = df['constructor_total_wins'] / df['constructor_total_races']

# Add names
drivers['fullName'] = drivers['forename'] + ' ' + drivers['surname']
df = df.merge(drivers[['driverId', 'fullName']], on='driverId', how='left')
df = df.merge(circuits[['circuitId', 'circuit_name', 'location', 'country']], on='circuitId', how='left')

# Filter high-performing drivers and popular circuits
avg_driver_win_rate = df['driver_win_rate'].mean()
avg_circuit_count = df['circuit_name'].value_counts().mean()
valid_drivers = df[df['driver_win_rate'] >= avg_driver_win_rate]['fullName'].unique()
valid_circuits = df['circuit_name'].value_counts()
valid_circuits = valid_circuits[valid_circuits >= avg_circuit_count].index.tolist()
df = df[df['fullName'].isin(valid_drivers) & df['circuit_name'].isin(valid_circuits)]

# Encoding
le_driver = LabelEncoder()
le_circuit = LabelEncoder()
df['driver_encoded'] = le_driver.fit_transform(df['fullName'])
df['circuit_encoded'] = le_circuit.fit_transform(df['circuit_name'])

# Model
features = ['grid', 'driver_win_rate', 'constructor_win_rate', 'driver_encoded', 'circuit_encoded']
X = df[features]
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🚀 Streamlit App
st.title("🏁 F1 Race Winner Predictor")

st.markdown("### Select Driver, Circuit & Grid Position")
selected_driver = st.selectbox("Choose a Driver", sorted(valid_drivers))
selected_circuit = st.selectbox("Choose a Circuit", sorted(valid_circuits))
grid_position = st.slider("Grid Position (1 = Pole)", min_value=1, max_value=20, value=1)

if selected_driver and selected_circuit:
    try:
        driver_encoded = le_driver.transform([selected_driver])[0]
        circuit_encoded = le_circuit.transform([selected_circuit])[0]
        driver_win_rate = df[df['fullName'] == selected_driver]['driver_win_rate'].mean()
        constructor_win_rate = df[df['fullName'] == selected_driver]['constructor_win_rate'].mean()

        input_data = pd.DataFrame({
            'grid': [grid_position],
            'driver_win_rate': [driver_win_rate],
            'constructor_win_rate': [constructor_win_rate],
            'driver_encoded': [driver_encoded],
            'circuit_encoded': [circuit_encoded]
        })

        win_prob = model.predict_proba(input_data)[0][1]

        st.success(f"🏆 {selected_driver} at {selected_circuit} from grid {grid_position} has a **{win_prob*100:.2f}%** chance of winning.")
    except:
        st.error("❌ Unable to process input. Check driver or circuit selection.")
