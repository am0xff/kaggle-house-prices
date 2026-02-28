from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"
TRAIN_PATH = DATA_RAW_DIR / "train.csv"
TEST_PATH = DATA_RAW_DIR / "test.csv"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"

RANDOM_SEED = 0xFF
CV_N_SPLITS = 5
