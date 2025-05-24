# F1 Race Winner Predictor

## Project Overview
The **F1 Race Winner Predictor** is a machine learning project designed to estimate the probability of a Formula 1 driver winning a race based on historical data. The model incorporates driver performance, constructor stats, circuit characteristics, and grid positions to make predictions.

This project includes:
- An **interactive Streamlit web app** (`ok.py`) where users can select a driver, circuit, and grid position to get a winning probability.
- A **standalone prediction script** (`prediction_script.py`) that runs the machine learning model directly.
- A `data/` folder containing all necessary CSV datasets for training and prediction.

---

## Features
- Historical race data integration (drivers, circuits, results, status, and races)
- Feature engineering to compute win rates for drivers and constructors
- Encoding of categorical features for ML model compatibility
- Random Forest classification to predict race winners
- User-friendly web interface using Streamlit

---

## Dataset Details
The `data/` folder should contain the following CSV files:

- `races.csv` — Information about races, including raceId, year, circuitId, date, etc.
- `results.csv` — Results of each race with positions and driver info
- `status.csv` — Status codes explaining race results
- `drivers.csv` — Driver details including names
- `circuits.csv` — Circuit information and locations

**Note:**  
If you don't have these files, you can download official F1 data from public datasets such as the [Ergast API](http://ergast.com/mrd/) or Kaggle’s F1 datasets.

---

## Installation

1. Clone this repository or download the source code.
2. Make sure Python 3.7+ is installed.
3. Navigate to the project directory in your terminal.
4. Install the required dependencies using:

```bash
pip install -r requirements.txt

