# TODO: delete unneeded imports, get XGboost to work with my H2O set up

import os
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
import psutil
import random
from sklearn.metrics import mean_absolute_percentage_error
import tqdm as tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, '../output/train_final_processed.csv')
TEST_PATH = os.path.join(BASE_DIR, '../output/test_final_processed.csv')
SUBMISSION_PATH = os.path.join(BASE_DIR, '../data/sample_submission.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Functions

def init_h2o():
    """Initialize H2O with system constraints."""
    pct_memory = 0.7
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    min_mem_size = round(pct_memory * available_memory_gb)
    port_no = random.randint(5555, 55555)
    os.environ["H2O_XGBOOST_MINIMUM_REQUIREMENTS"] = "true"
    h2o.init(port=port_no, strict_version_check=False, min_mem_size_GB=4, nthreads=4)
    h2o.download_all_logs(dirname="./h2o_logs")


def load_data():
    """Load preprocessed data."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return h2o.H2OFrame(train), h2o.H2OFrame(test)

def train_automl(train, y_col='num_sold', runtime_secs=3600):
    """Train AutoML model."""
    print("Starting H2O AutoML...")
    x_cols = [col for col in train.columns if col != y_col]
    from tqdm import tqdm
    pbar = tqdm(total=100, desc="💪🏼Time for a training montage! 💪🏼🏋🏽‍♂️")
    try:
        aml = H2OAutoML(max_runtime_secs=runtime_secs, stopping_metric='mae', seed=42)
        aml.train(x=x_cols, y=y_col, training_frame=train)
        pbar.update(80)

        leaderboard = aml.leaderboard
        # but make it fun
        print("✨🏆Leaderboard has been generated!🏆🪄")
        print(leaderboard)
        pbar.update(20)
    finally:
        pbar.close()

    return aml

def evaluate_model(aml, test, y_col='num_sold'):
    """Evaluate the best model."""
    predictions = aml.leader.predict(test)
    predictions = predictions.as_data_frame()
    test = test.as_data_frame()

    # MAPE calculation
    # since MAPE is the target LB metric for the kaggle competition
    y_true = test[y_col]
    y_pred = predictions['predict']
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"MAPE: {mape:.2f}%")
    return predictions

def save_submission(predictions):
    """Save submission file."""
    sample = pd.read_csv(SUBMISSION_PATH)
    submission = pd.DataFrame({'id': sample['id'], 'num_sold': predictions['predict']})
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"🌈🦄Success! Submission  saved to {submission_path}")
    return submission_path

# 🪿
def submit_to_kaggle(submission_path, model_name):
    """Submit predictions to Kaggle."""
    os.system(f'kaggle competitions submit -c playground-series-s5e1 -f {submission_path} -m "{model_name}"')

if __name__ == '__main__':
    init_h2o()
    train, test = load_data()

    # Train AutoML
    aml = train_automl(train)

    from h2o.explanation import explain

    explain_output = explain(aml.leader, test)

    # Evaluate and predict
    predictions = evaluate_model(aml, test)

    # Save predictions and submit
    submission_path = save_submission(predictions)
    submit_to_kaggle(submission_path, model_name=aml.leader.model_id)

    # Turn off the faucet. 🚰 Shut down the H20.
    h2o.cluster().shutdown()
