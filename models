import pandas as pd
import numpy as np
import os
import joblib
from colorama import Fore, Style, init
from tabulate import tabulate
from collections import defaultdict

# Regression Models (Teams)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Classification Models (Games)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Initialize colorama
init(autoreset=True)

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM_DATA_PATH = os.path.join(BASE_DIR, 'data', 'teams_data.xlsx')
MATCH_DATA_PATH = os.path.join(BASE_DIR, 'data', 'matches_data.xlsx')

# Folder to save trained models
MODEL_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def print_header(text):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*70}")
    print(f"{Fore.CYAN}{Style.BRIGHT}  {text}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*70}")

# Initialize result holders
lr_team_r2 = 0.0
dt_team_r2 = 0.0
log_game_acc = 0.0
dt_game_acc = 0.0

# =====================================================
# TEAM DATA (REGRESSION)
# =====================================================
print_header("TEAM PERFORMANCE ANALYSIS (REGRESSION MODELS)")

if os.path.exists(TEAM_DATA_PATH):

    df_team = pd.read_excel(TEAM_DATA_PATH)

    # DATA PREVIEW
    print(f"{Fore.YELLOW}--- RAW TEAM DATA PREVIEW ---")
    print(df_team.head(10))
    print(f"{Fore.YELLOW}--- DATA STRUCTURE ---")
    df_team.info()
    print(f"{Fore.YELLOW}-----------------------------")

    df_team.columns = df_team.columns.str.strip().str.upper()

features_team = [
    'RUNS_SCORED',
    'RUNS_CONCEDED',
    'RUN_DIFFERENTIAL',
    'GAMES_PLAYED'
]

target_team = 'WINS'

if all(col in df_team.columns for col in features_team + [target_team]):

    # FEATURE & TARGET SEPARATION
    X_team = df_team[features_team].dropna()
    y_team = df_team[target_team].loc[X_team.index]

    # Train-test split
    X_train_team, X_test_team, y_train_team, y_test_team = train_test_split(
        X_team,
        y_team,
        test_size=0.2,
        random_state=42
    )

    # Linear Regression
    lr_team_model = LinearRegression()
    lr_team_model.fit(X_train_team, y_train_team)

    lr_team_predictions = lr_team_model.predict(X_test_team)
    lr_team_r2 = r2_score(y_test_team, lr_team_predictions)

    # Decision Tree
    dt_team_model = DecisionTreeRegressor(random_state=42)
    dt_team_model.fit(X_train_team, y_train_team)

    dt_team_predictions = dt_team_model.predict(X_test_team)
    dt_team_r2 = r2_score(y_test_team, dt_team_predictions)

print(f"{Fore.GREEN}✓ Team Data Successfully Split into Training & Testing Sets")

# =====================================================
# TEAM DATA RANDOM FOREST
# =====================================================
print_header("TEAM PERFORMANCE ANALYSIS (RANDOM FOREST REGRESSOR)")

if os.path.exists(TEAM_DATA_PATH):

    df_team = pd.read_excel(TEAM_DATA_PATH)
    df_team.columns = df_team.columns.str.strip().str.upper()

    # Create per-game features
    df_team['RUNS_SCORED_PER_GAME'] = df_team['RUNS_SCORED'] / df_team['GAMES_PLAYED']
    df_team['RUNS_CONCEDED_PER_GAME'] = df_team['RUNS_CONCEDED'] / df_team['GAMES_PLAYED']
    df_team['RUN_DIFFERENTIAL_PER_GAME'] = df_team['RUN_DIFFERENTIAL'] / df_team['GAMES_PLAYED']

    features_team = [
        'RUNS_SCORED_PER_GAME',
        'RUNS_CONCEDED_PER_GAME',
        'RUN_DIFFERENTIAL_PER_GAME',
        'WINNING_PERCENTAGE'
    ]
    target_team = 'WINS'

    X_team = df_team[features_team]
    y_team = df_team[target_team]

    # Train-test split
    X_train_team, X_test_team, y_train_team, y_test_team = train_test_split(
        X_team, y_team, test_size=0.2, random_state=42
    )

    # Random Forest Regressor
    rf_team_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    print(f"{Fore.BLUE}Training Random Forest Regressor (Teams)...")
    rf_team_model.fit(X_train_team, y_train_team)

    # Predictions
    rf_team_predictions = rf_team_model.predict(X_test_team)

    # Metrics
    rf_team_r2 = r2_score(y_test_team, rf_team_predictions)
    rf_team_mae = mean_absolute_error(y_test_team, rf_team_predictions)
    rf_team_rmse = np.sqrt(mean_squared_error(y_test_team, rf_team_predictions))

    print(f"{Fore.MAGENTA}Random Forest R² (Teams): {rf_team_r2:.4f}")
    print(f"{Fore.MAGENTA}MAE: {rf_team_mae:.2f}, RMSE: {rf_team_rmse:.2f}")

    # Save model
    joblib.dump(rf_team_model, os.path.join(MODEL_DIR, "random_forest_team.pkl"))
    print(f"{Fore.GREEN}✓ Random Forest Team model saved successfully")

else:
    print(f"{Fore.RED}Team data file not found at {TEAM_DATA_PATH}")

# =====================================================
# MATCH DATA CLASSIFICATION
# =====================================================
print_header("MATCH TREND ANALYSIS (CLASSIFICATION MODELS)")

if os.path.exists(MATCH_DATA_PATH):

    df_match = pd.read_excel(MATCH_DATA_PATH)

    # DATA PREVIEW
    print(f"{Fore.YELLOW}--- RAW MATCH DATA PREVIEW ---")
    print(df_match.head(10))
    print(f"{Fore.YELLOW}--- DATA STRUCTURE ---")
    df_match.info()
    print(f"{Fore.YELLOW}-----------------------------")

    df_match.columns = df_match.columns.str.strip().str.upper()

# =====================================================
# MERGE TEAM STATS INTO MATCH DATA
# =====================================================

if os.path.exists(TEAM_DATA_PATH):

    # Reload clean team data
    df_team = pd.read_excel(TEAM_DATA_PATH)
    df_team.columns = df_team.columns.str.strip().str.upper()

    # Ensure TEAM column exists
    if 'TEAM' in df_team.columns:

        # CLEAN TEAM NAMES BEFORE MERGING
        df_match["HOME TEAM"] = df_match["HOME TEAM"].astype(str).str.strip().str.title()
        df_match["AWAY TEAM"] = df_match["AWAY TEAM"].astype(str).str.strip().str.title()
        df_team["TEAM"] = df_team["TEAM"].astype(str).str.strip().str.title()

        # Add prefixes to distinguish Home vs Away stats
        df_home = df_team.add_prefix("HOME_")
        df_away = df_team.add_prefix("AWAY_")

        # Merge Home team stats
        df_match = df_match.merge(
            df_home,
            left_on="HOME TEAM",
            right_on="HOME_TEAM",
            how="left"
        )

        # Merge Away team stats
        df_match = df_match.merge(
            df_away,
            left_on="AWAY TEAM",
            right_on="AWAY_TEAM",
            how="left"
        )

        print(f"{Fore.GREEN}✓ Team statistics merged into match dataset")

        # DEBUG CHECK
        print("\nDEBUG CHECK:")
        print("Total Rows:", len(df_match))
        print("Missing HOME stats:", df_match["HOME_WINNING_PERCENTAGE"].isna().sum())
        print("Missing AWAY stats:", df_match["AWAY_WINNING_PERCENTAGE"].isna().sum())

    else:
        print(f"{Fore.RED}TEAM column missing in teams_data.xlsx")

# REAL TARGET: Based on actual match scores
df_match["HOME_WIN"] = (
    df_match["HOME_SCORE"] > df_match["AWAY_SCORE"]
).astype(int)

print(f"{Fore.GREEN}✓ HOME_WIN target created using real match scores")

# =====================================================
# ELO RATING SYSTEM
# =====================================================

print_header("ELO RATING CALCULATION")

K_FACTOR = 32
BASE_RATING = 1500

elo_ratings = defaultdict(lambda: BASE_RATING)

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

for index, row in df_match.iterrows():

    home_team = row["HOME TEAM"]
    away_team = row["AWAY TEAM"]

    home_rating = elo_ratings[home_team]
    away_rating = elo_ratings[away_team]

    exp_home = expected_score(home_rating, away_rating)
    exp_away = expected_score(away_rating, home_rating)

    actual_home = row["HOME_WIN"]
    actual_away = 1 - actual_home

    elo_ratings[home_team] += K_FACTOR * (actual_home - exp_home)
    elo_ratings[away_team] += K_FACTOR * (actual_away - exp_away)

df_match["HOME_ELO"] = df_match["HOME TEAM"].map(elo_ratings)
df_match["AWAY_ELO"] = df_match["AWAY TEAM"].map(elo_ratings)

print(f"{Fore.GREEN}✓ ELO ratings successfully calculated")

# MATCH DATA + IMPUTATION
features_match = [
    "HOME_WINNING_PERCENTAGE",
    "AWAY_WINNING_PERCENTAGE",
    "HOME_RUN_DIFFERENTIAL",
    "AWAY_RUN_DIFFERENTIAL",
    "HOME_GAMES_PLAYED",
    "AWAY_GAMES_PLAYED",
    "HOME_ELO",
    "AWAY_ELO"
]

target_match = "HOME_WIN"

df_match = df_match[features_match + [target_match]].copy()

# Handle missing values
imputer = SimpleImputer(strategy="median")

X_match = imputer.fit_transform(df_match[features_match])
y_match = df_match[target_match].values

# Interaction feature

X_match = np.column_stack([
    X_match,
    X_match[:,0] * X_match[:,1]
])
# run differential matchup feature
X_match = np.column_stack([
    X_match,
    X_match[:,2] - X_match[:,3]
])
# Train-test split
X_train_match, X_test_match, y_train_match, y_test_match = train_test_split(
    X_match, y_match,
    test_size=0.2,
    random_state=42
)

print(f"{Fore.GREEN}✓ Match Data Successfully Split into Training & Testing Sets")

# RANDOM FOREST (GAMES)
print_header("RANDOM FOREST - GAMES")

from sklearn.ensemble import RandomForestClassifier

rf_game_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_game_model.fit(X_train_match, y_train_match)

rf_game_predictions = rf_game_model.predict(X_test_match)
rf_game_probabilities = rf_game_model.predict_proba(X_test_match)[:,1]
log_game_acc = accuracy_score(y_test_match, rf_game_predictions)

print(f"{Fore.MAGENTA}Random Forest Accuracy (Games): {log_game_acc:.4f}")
print("\nSample Win Probabilities:")
for i in range(5):
    print(f"Match {i+1} Home Win Probability: {rf_game_probabilities[i]:.2%}")

# OPTIMIZED DECISION TREE CLASSIFIER
print_header("OPTIMIZED DECISION TREE CLASSIFIER - GAMES")

dt_param_grid = {
    "max_depth": [3, 5, 8, 12],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

dt_grid.fit(X_match, y_match)

best_dt_model = dt_grid.best_estimator_

dt_game_predictions = best_dt_model.predict(X_test_match)

dt_game_acc = accuracy_score(y_test_match, dt_game_predictions)

print(f"{Fore.MAGENTA}Optimized Decision Tree Accuracy (Games): {dt_game_acc:.4f}")

print("\nBest Decision Tree Parameters:")
print(dt_grid.best_params_)

print("\nClassification Report:")
print(classification_report(y_test_match, dt_game_predictions))

# =====================================================
# SUMMARY TABLE
# =====================================================
print_header("SUMMARY")

summary_data = [
    ["Assignment Target", "Linear/Logistic", "Decision Tree"],
    ["Team Performance (R²)", f"{lr_team_r2:.4f}", f"{dt_team_r2:.4f}"],
    ["Home Win Prediction (Accuracy)", f"{log_game_acc:.4f}", f"{dt_game_acc:.4f}"]
]

print(tabulate(summary_data, headers="firstrow", tablefmt="fancy_grid"))

print(f"\n{Fore.CYAN}{Style.BRIGHT}HEAD-TO-HEAD WINNERS")
print(f"Teams Model Winner: {'Linear Regression' if lr_team_r2 > dt_team_r2 else 'Decision Tree'}")
print(f"Games Model Winner: {'Random Forest' if log_game_acc > dt_game_acc else 'Decision Tree'}")

print(f"\n{Fore.GREEN}✅ All models trained and tested separately for Teams and Games.")

# =====================================================
# SAVE TRAINED MODELS
# =====================================================

MODEL_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

joblib.dump(lr_team_model, os.path.join(MODEL_DIR, "linear_regression_team.pkl"))
joblib.dump(dt_team_model, os.path.join(MODEL_DIR, "decision_tree_team.pkl"))
joblib.dump(rf_game_model, os.path.join(MODEL_DIR, "random_forest_game.pkl"))
joblib.dump(best_dt_model, os.path.join(MODEL_DIR, "decision_tree_game.pkl"))

print(f"{Fore.GREEN}✓ Models saved successfully for Flask integration")

def train_models():
    import pandas as pd
    import joblib
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    players = pd.read_excel("data/players_data.xlsx")
    teams = pd.read_excel("data/teams_data.xlsx")

    # Player model
    X_player = players[['HITS','AT_BAT','HOMERUNS','STRIKE_OUTS']]
    y_player = players['BATTING_AVG']
    player_model = LinearRegression()
    player_model.fit(X_player, y_player)
    joblib.dump(player_model, "player_lr_model.pkl")

    # Team model
    X_team = teams[['HITS','HOMERUNS','STRIKE_OUTS']]
    y_team = teams['WINS']
    team_model = DecisionTreeRegressor()
    team_model.fit(X_team, y_team)
    joblib.dump(team_model, "team_dt_model.pkl")

    print("Models trained and saved.")

# Save only trained models
joblib.dump(rf_team_model, os.path.join(MODEL_DIR, "random_forest_team.pkl"))
joblib.dump(rf_game_model, os.path.join(MODEL_DIR, "random_forest_game.pkl"))
joblib.dump(best_dt_model, os.path.join(MODEL_DIR, "decision_tree_game.pkl"))
