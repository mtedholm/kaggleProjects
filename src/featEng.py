import os
import pandas as pd
import numpy as np
import wbdata
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import holidays
import ephem
import pytz
import datetime
from sklearn import preprocessing

tqdm.pandas()

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, '../data/test.csv')
TRAIN_PATH = os.path.join(BASE_DIR, '../data/train.csv')
SUBMISSION_PATH = os.path.join(BASE_DIR, '../data/sample_submission.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mappings
TIMEZONES = {
    'Canada': 'America/Toronto',
    'Finland': 'Europe/Helsinki',
    'Italy': 'Europe/Rome',
    'Norway': 'Europe/Oslo',
    'Singapore': 'Asia/Singapore',
    'Kenya': 'Africa/Nairobi'
}

CONTINENTS = {
    "Canada": "North America",
    "Italy": "Europe",
    "Norway": "Europe",
    "Finland": "Europe",
    "Kenya": "Africa",
    "Singapore": "Asia",
}

# Functions
def load_data():
    """Load train and test datasets."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def preprocess_date(df):
    """Preprocess the date column and extract date-based features."""
    tqdm.pandas(desc="Adding time-based features")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Weekday'] = df['date'].dt.dayofweek
    df['Quarter'] = df['date'].dt.quarter
    df['Continent'] = df['country'].map(CONTINENTS)
    df['is_weekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Add lag and rolling average only if `num_sold` exists (train dataset)
    # impute an effectively 0 value for NaN
    if 'num_sold' in df.columns:
        df['num_sold'] = df['num_sold'].fillna(0.00000001)
        df['lag_1'] = df['num_sold'].shift(1)
        df['rolling_avg_7'] = df['num_sold'].rolling(7).mean()
        print(f"Lag and rolling averages added:\n{df[['lag_1', 'rolling_avg_7']].head()}")
    else:
        print("Skipping lag and rolling average as 'num_sold' is not in columns.")

    return df

# now redundant since i refactored preprocess_data to just do everything
def add_continent(df):
    """Add continent information."""
    df['Continent'] = df['country'].map(CONTINENTS)
    return df

# now redundant since i refactored preprocess_data to just do everything
def add_is_weekend(df):
    """Add weekend indicator."""
    df['is_weekend'] = df['Weekday'].isin([5, 6]).astype(int)
    return df

# now redundant since i refactored preprocess_data to just do everything
def add_features(df):
    """Add lag, rolling averages, and sinusoidal encodings."""
    tqdm.pandas(desc="Adding time-based features")
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Add lag and rolling average only if `num_sold` exists (train dataset)
    if 'num_sold' in df.columns:
        df['num_sold'] = df['num_sold'].fillna(0.000001)
        df['lag_1'] = df['num_sold'].shift(1)
        df['rolling_avg_7'] = df['num_sold'].rolling(7).mean()
        print(f"Lag and rolling averages added:\n{df[['lag_1', 'rolling_avg_7']].head()}")
    else:
        print("Skipping lag and rolling average as 'num_sold' is not in columns.")

    return df



def encode_categorical(df):
    """
    Encode all object-type columns in the DataFrame using LabelEncoder.
    """
    tqdm.pandas(desc="Encoding categorical values")
    label_encoder = preprocessing.LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col].astype(str))
    return df


# Still a WIP:
# TODO: refactor to just use worldbank API instead of the python wrapper
def fetch_gdp_data(countries, years):
    """Fetch GDP data for the specified countries and years."""
    tqdm.pandas(desc="Retrieving GDP data")

    country_code_map = {
        'Canada': 'CAN',
        'Finland': 'FIN',
        'Italy': 'ITA',
        'Norway': 'NOR',
        'Singapore': 'SGP',
        'Kenya': 'KEN'
    }
    # Just some starter indicators that could potentially be slightly relevant
    indicators = {
        'Poverty_Headcount_Vul4to10': '1.2.HCount.Vul4to10',
        'GDP_Per_Capita': 'NY.GDP.PCAP.CD',
        'Life_Expectancy': 'SP.DYN.LE00.IN'
    }

    gdp_data = None  # Initialize this to handle errors gracefully

    try:
        # Fetch data for all years
        gdp_data = wbdata.get_dataframe(
            indicators=indicators,
            country=['CAN', 'FIN', 'ITA', 'NOR', 'SGP', 'KEN']
        )

        # Reset the index to make 'date' and 'country' columns
        gdp_data = gdp_data.reset_index()

        # Filter for the desired years (2013–2017)
        gdp_data['date'] = pd.to_datetime(gdp_data['date'])
        gdp_data = gdp_data[(gdp_data['date'] >= '2013-01-01') & (gdp_data['date'] <= '2017-12-31')]

        # Rename columns for clarity
        gdp_data = gdp_data.rename(columns={
            'date': 'Year',
            'Poverty_Headcount_Vul4to10': 'Poverty_Vulnerable',
            'GDP_Per_Capita': 'GDP_Per_Capita',
            'Life_Expectancy': 'Life_Expectancy'
        })
        gdp_data['Year'] = gdp_data['Year'].dt.year

        #print(gdp_data.head())

    except Exception as e:
        print(f"Error fetching indicators: {e}")

    return gdp_data

# adds the GDP data to the original df
def add_gdp(df, gdp_data):
    """Merge GDP data into the dataframe based on country and Year."""
    tqdm.pandas(desc="Merging GDP data")
    df = pd.merge(df, gdp_data, on=['country', 'Year'], how='left')
    return df

# just for fun, why not. Maybe people buy more holographic Kaggle stickers during full moons.
def add_moon_phase(df):
    """Add moon phase and category features."""
    def moon_phase(date):
        moon = ephem.Moon(date)
        return moon.phase

    # TODO: refactor this fxn out since it all gets label encoded anyway

    def moon_phase_category(phase):
        if 0 <= phase < 10 or 90 <= phase <= 100:
            return 'New Moon'
        elif 10 <= phase < 40:
            return 'Waxing Crescent'
        elif 40 <= phase < 60:
            return 'First Quarter'
        elif 60 <= phase < 90:
            return 'Waxing Gibbous'
        else:
            return 'Full Moon'

    # biggest accomplishment today: adding emoji to the TQDM bar
    tqdm.pandas(desc="Calculating moon phases 🌕🌖🌗🌘🌑🌒")
    df['moon_phase'] = df['date'].progress_apply(moon_phase)
    df['moon_phase_category'] = df['moon_phase'].progress_apply(moon_phase_category)
    return df

def save_data(train, test):
    """Save processed train and test datasets."""
    train.to_csv(os.path.join(OUTPUT_DIR, 'train_final_processed.csv'), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, 'test_final_processed.csv'), index=False)

# Main Pipeline
def main():
    train, test = load_data()
    # Preprocessing
    train = preprocess_date(train)
    test = preprocess_date(test)

    train = add_moon_phase(train)
    test = add_moon_phase(test)

    # Encode categorical variables

    train = encode_categorical(train)
    test = encode_categorical(test)

    # Save processed data
    save_data(train, test)
    print("Success! Preprocessing and feature engineering completed.")

if __name__ == '__main__':
    main()
