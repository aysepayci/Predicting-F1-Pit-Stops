"""Central config — paths and constants. Import this everywhere."""
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_EXT    = ROOT / "data" / "external"
REPORTS     = ROOT / "reports"
SUBMISSIONS = ROOT / "submissions"
MODELS_DIR  = ROOT / "models"

TRAIN_FILE  = DATA_RAW / "train.csv"
TEST_FILE   = DATA_RAW / "test.csv"

TARGET       = "PitNextLap"
GROUP_COL    = "race_uid"   # Race + "_" + str(Year)
ID_COL       = "id"
RANDOM_SEED  = 42
N_FOLDS      = 5

NUMERIC_COLS = [
    "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "PitStop",
]
CATEGORICAL_COLS = ["Driver", "Compound", "Race"]
DROP_COLS        = [ID_COL, TARGET, "Year", GROUP_COL]