import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import ephem
import holidays
import math
import os
import psutil
import pytz
import random
import re
import requests
import time
from tqdm import tqdm



from collections import defaultdict

import pytz
from datetime import datetime

import warnings

tqdm.pandas()
# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# This is a preprocessing script to add features, etc to the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_PATH = os.path.join(BASE_DIR, '../data/test.csv')
TRAIN_PATH = os.path.join(BASE_DIR,'../data/train.csv')
SUBMISSION_PATH = os.path.join(BASE_DIR,'../data/sample_submission.csv')

# mappings

# tz for moon phase
TIMEZONES = {
    'Canada': 'America/Toronto',
    'Finland': 'Europe/Helsinki',
    'Italy': 'Europe/Rome',
    'Norway': 'Europe/Oslo',
    'Singapore': 'Asia/Singapore',
    'Kenya': 'Africa/Nairobi'
}

# Country-Continent Mapping
CONTINENTS = {
    "Canada": "North America",
    "Italy": "Europe",
    "Norway": "Europe",
    "Finland": "Europe",
    "Kenya": "Africa",
    "Singapore": "Asia",
}


def load_data():
  """Load train and test datasets."""
  train = pd.read_csv(TRAIN_PATH)
  test = pd.read_csv(TEST_PATH)
  return train, test

def preprocess_date(df):
    """Preprocess date column and extract features."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Weekday'] = df['date'].dt.dayofweek
    df['Quarter'] = df['date'].dt.quarter
    return df


def add_continent(df):
  """Add continent based on country."""
  df['Continent'] = df['country'].map(CONTINENTS)
  return df

def add_is_weekend(df):
    """Add a feature indicating if the date is a weekend."""
    df['is_weekend'] = df['Weekday'].isin([5, 6]).astype(int)
    return df

def impute_num_sold(df):
  """Impute 'num_sold' column for NaN values."""
  if 'num_sold' in df.columns:
    df['num_sold'].fillna(0.00000000001, inplace=True)
  return df

def add_is_holiday(df):
    """Add a feature indicating if the date is a holiday."""

    def is_holiday(row):
      try:
        hols = holidays.country_holidays(row['country'], years=row['Year'])
        return 1 if row['date'] in hols else 0
      except KeyError:
        return 0

    df['is_holiday'] = df.progress_apply(is_holiday, axis=1)
    return df


def add_days_to_holiday(df):
  """Add days to next and previous holidays."""
  min_year = df['date'].dt.year.min()
  max_year = df['date'].dt.year.max()
  years = list(range(min_year - 1, max_year + 1))

  holiday_objects = {
    'Canada': holidays.country_holidays('CA', years=years),
    'Finland': holidays.country_holidays('FI', years=years),
    'Italy': holidays.country_holidays('IT', years=years),
    'Norway': holidays.country_holidays('NO', years=years),
    'Singapore': holidays.country_holidays('SG', years=years),
    'Kenya': holidays.country_holidays('KE', years=years)
  }

  def days_to_holiday(row):
    country = row['country']
    date_obj = row['date'].date()
    holidays_for_country = holiday_objects.get(country, [])

    if not holidays_for_country:
      return None, None

    next_holidays = [holiday for holiday in holidays_for_country if holiday >= date_obj]
    prev_holidays = [holiday for holiday in holidays_for_country if holiday <= date_obj]

    days_to_next = (min(next_holidays) - date_obj).days if next_holidays else float('inf')
    days_from_prev = (date_obj - max(prev_holidays)).days if prev_holidays else float('inf')

    return days_to_next, days_from_prev

  df[['days_to_next_holiday', 'days_from_prev_holiday']] = df.progress_apply(
    lambda row: pd.Series(days_to_holiday(row)), axis=1
  )
  return df


def add_moon_phase(df):
  """Add moon phase and category features."""

  def moon_phase(date):
    moon = ephem.Moon(date)
    return moon.phase

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

  def local_moon_phase(row):
    timezone = pytz.timezone(TIMEZONES.get(row['country'], 'UTC'))
    local_date = row['date'].tz_localize('UTC').tz_convert(timezone)
    return moon_phase(local_date)

  df['moon_phase'] = df['date'].progress_apply(moon_phase)
  df['moon_phase_category'] = df['moon_phase'].progress_apply(moon_phase_category)
  df['local_moon_phase'] = df.progress_apply(local_moon_phase, axis=1)
  return df

def main():
    """Main function to preprocess data."""
    output_dir = os.path.join(BASE_DIR, '../data')
    os.makedirs(output_dir, exist_ok=True)

    train, test = load_data()
    train = preprocess_date(train)
    test = preprocess_date(test)

    train = add_continent(train)
    test = add_continent(test)

    train = add_is_weekend(train)
    test = add_is_weekend(test)

    train = impute_num_sold(train)

    train = add_is_holiday(train)
    test = add_is_holiday(test)

    train = add_days_to_holiday(train)
    test = add_days_to_holiday(test)

    train = add_moon_phase(train)
    test = add_moon_phase(test)

    # Save processed data (optional)
    train.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)

print("Preprocessing completed successfully.")


if __name__ == "__main__":
  main()
