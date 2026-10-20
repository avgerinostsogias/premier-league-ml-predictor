from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from data_collection import load_data
from features import create_features


def create_model(c_value=1.0):
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=c_value,
            ),
        ),
    ])


def apply_temperature(probabilities, temperature):
    probabilities = np.clip(
        probabilities,
        1e-15,
        1,
    )

    log_probabilities = np.log(
        probabilities
    )

    scaled_log_probabilities = (
        log_probabilities / temperature
    )

    # Numerical stability
    scaled_log_probabilities -= (
        scaled_log_probabilities.max(
            axis=1,
            keepdims=True,
        )
    )

    exp_probabilities = np.exp(
        scaled_log_probabilities
    )

    calibrated_probabilities = (
        exp_probabilities
        / exp_probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    return calibrated_probabilities


def multiclass_brier_score(
    y_true,
    probabilities,
    classes,
):
    y_true_array = np.array(y_true)

    one_hot = np.zeros_like(
        probabilities
    )

    for i, class_name in enumerate(classes):
        one_hot[:, i] = (
            y_true_array == class_name
        ).astype(int)

    squared_errors = (
        probabilities - one_hot
    ) ** 2

    return np.mean(
        np.sum(
            squared_errors,
            axis=1,
        )
    )


# ----------------------------------
# Load data and create features
# ----------------------------------

df = load_data()
df = create_features(df)

features = [
    "HomePointsAvgLast5",
    "AwayPointsAvgLast5",

    "HomeGoalsAvgLast5",
    "AwayGoalsAvgLast5",

    "HomeConcededAvgLast5",
    "AwayConcededAvgLast5",

    "HomePointsAvgHomeLast5",
    "AwayPointsAvgAwayLast5",

    "HomeShotAccuracyLast5",
    "AwayShotAccuracyLast5",

    "HomeHistoricalStrength",
    "AwayHistoricalStrength",

    "HomePromoted",
    "AwayPromoted",

    "HomeNoPLHistory",
    "AwayNoPLHistory",

    "HomeRecentMatchesAvailable",
    "AwayRecentMatchesAvailable",

    "EloDifference"
]

model_seasons = [
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]


# ----------------------------------
# Walk-forward evaluation
# Uses default C=1.0 as the untuned benchmark
# ----------------------------------

walk_forward_results = []

for test_index in range(1, len(model_seasons)):
    test_season = model_seasons[test_index]
    train_seasons = model_seasons[:test_index]

    train_df = df[
        df["Season"].isin(train_seasons)
    ].copy()

    walk_test_df = df[
        df["Season"] == test_season
    ].copy()

    x_train = train_df[features]
    y_train = train_df["FTR"]

    x_test = walk_test_df[features]
    y_test = walk_test_df["FTR"]

    model = create_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    classes = model.named_steps[
        "logistic_regression"
    ].classes_

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    current_log_loss = log_loss(
        y_test,
        probabilities,
        labels=classes,
    )

    current_brier = multiclass_brier_score(
        y_test,
        probabilities,
        classes,
    )

    # Baseline: predict the most common training result
    majority_class = (
        y_train.value_counts().idxmax()
    )

    baseline_predictions = np.full(
        len(y_test),
        majority_class,
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    walk_forward_results.append({
        "TestSeason": test_season,
        "TrainSeasons": len(train_seasons),
        "TrainingMatches": len(train_df),
        "TestingMatches": len(walk_test_df),
        "Accuracy": accuracy,
        "BaselineAccuracy": baseline_accuracy,
        "MacroF1": macro_f1,
        "LogLoss": current_log_loss,
        "BrierScore": current_brier,
    })

    print("\n-------------------------")
    print("Test season:", test_season)
    print("Training seasons:", train_seasons)
    print("Training matches:", len(train_df))
    print("Accuracy:", accuracy)
    print("Baseline:", baseline_accuracy)
    print("Macro F1:", macro_f1)
    print("Log Loss:", current_log_loss)
    print("Brier Score:", current_brier)


# ----------------------------------
# Tune Logistic Regression C
# Only previous seasons are used for tuning
# ----------------------------------

c_values = [
    0.01,
    0.03,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]

validation_test_seasons = [
    "2023-24",
    "2024-25",
]

tuning_results = []

for c_value in c_values:
    fold_accuracies = []
    fold_f1_scores = []
    fold_log_losses = []
    fold_brier_scores = []

    for test_season in validation_test_seasons:
        test_index = model_seasons.index(
            test_season
        )

        train_seasons = model_seasons[
            :test_index
        ]

        train_df = df[
            df["Season"].isin(train_seasons)
        ].copy()

        validation_df = df[
            df["Season"] == test_season
        ].copy()

        x_train = train_df[features]
        y_train = train_df["FTR"]

        x_validation = validation_df[
            features
        ]
        y_validation = validation_df[
            "FTR"
        ]

        model = create_model(
            c_value
        )
        model.fit(
            x_train,
            y_train,
        )

        predictions = model.predict(
            x_validation
        )
        probabilities = model.predict_proba(
            x_validation
        )

        classes = model.named_steps[
            "logistic_regression"
        ].classes_

        fold_accuracies.append(
            accuracy_score(
                y_validation,
                predictions,
            )
        )

        fold_f1_scores.append(
            f1_score(
                y_validation,
                predictions,
                average="macro",
            )
        )

        fold_log_losses.append(
            log_loss(
                y_validation,
                probabilities,
                labels=classes,
            )
        )

        fold_brier_scores.append(
            multiclass_brier_score(
                y_validation,
                probabilities,
                classes,
            )
        )

    tuning_results.append({
        "C": c_value,
        "AverageAccuracy": np.mean(
            fold_accuracies
        ),
        "AverageMacroF1": np.mean(
            fold_f1_scores
        ),
        "AverageLogLoss": np.mean(
            fold_log_losses
        ),
        "AverageBrier": np.mean(
            fold_brier_scores
        ),
    })


tuning_df = pd.DataFrame(
    tuning_results
)

tuning_df = tuning_df.sort_values(
    [
        "AverageAccuracy",
        "AverageLogLoss",
    ],
    ascending=[
        False,
        True,
    ],
)

print("\nC TUNING RESULTS")
print(
    tuning_df.to_string(
        index=False
    )
)

best_c = tuning_df.iloc[0]["C"]
print("\nBest C:", best_c)


# ----------------------------------
# Probability calibration
# Train: 2021-22 to 2023-24
# Calibration: 2024-25
# ----------------------------------

model_train_df = df[
    df["Season"].isin([
        "2021-22",
        "2022-23",
        "2023-24",
    ])
].copy()

calibration_df = df[
    df["Season"] == "2024-25"
].copy()

x_model_train = model_train_df[
    features
]
y_model_train = model_train_df[
    "FTR"
]

x_calibration = calibration_df[
    features
]
y_calibration = calibration_df[
    "FTR"
]

# Important: calibration model uses the tuned C
temporary_model = create_model(
    best_c
)

temporary_model.fit(
    x_model_train,
    y_model_train,
)

calibration_probabilities = (
    temporary_model.predict_proba(
        x_calibration
    )
)

classes = temporary_model.named_steps[
    "logistic_regression"
].classes_

best_temperature = None
best_calibration_loss = float("inf")

for temperature in np.arange(
    0.5,
    3.01,
    0.05,
):
    calibrated_probabilities = (
        apply_temperature(
            calibration_probabilities,
            temperature,
        )
    )

    current_loss = log_loss(
        y_calibration,
        calibrated_probabilities,
        labels=classes,
    )

    if current_loss < best_calibration_loss:
        best_calibration_loss = (
            current_loss
        )
        best_temperature = temperature

raw_calibration_loss = log_loss(
    y_calibration,
    calibration_probabilities,
    labels=classes,
)

print(
    "Raw calibration Log Loss:",
    raw_calibration_loss,
)
print(
    "Best Temperature:",
    best_temperature,
)
print(
    "Calibrated Log Loss:",
    best_calibration_loss,
)


# ----------------------------------
# Walk-forward summary
# ----------------------------------

walk_forward_df = pd.DataFrame(
    walk_forward_results
)

print("\n\nWALK-FORWARD RESULTS")
print(
    walk_forward_df.to_string(
        index=False
    )
)

print("\nAVERAGE RESULTS")
print(
    "Average Accuracy:",
    walk_forward_df["Accuracy"].mean(),
)
print(
    "Average Baseline:",
    walk_forward_df[
        "BaselineAccuracy"
    ].mean(),
)
print(
    "Average Macro F1:",
    walk_forward_df[
        "MacroF1"
    ].mean(),
)
print(
    "Average Log Loss:",
    walk_forward_df[
        "LogLoss"
    ].mean(),
)
print(
    "Average Brier:",
    walk_forward_df[
        "BrierScore"
    ].mean(),
)


# ----------------------------------
# Final model
# Train: 2021-22 to 2024-25
# Test: 2025-26
# ----------------------------------

train_df = df[
    df["Season"].isin([
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
    ])
].copy()

test_df = df[
    df["Season"] == "2025-26"
].copy()

x_train = train_df[features]
y_train = train_df["FTR"]

x_test = test_df[features]
y_test = test_df["FTR"]

print(
    "Number of features:",
    len(features),
)
print(
    "Best C used in final model:",
    best_c,
)

# Important: final model also uses the tuned C
model = create_model(
    best_c
)

model.fit(
    x_train,
    y_train,
)

predictions = model.predict(
    x_test
)

raw_probabilities = model.predict_proba(
    x_test
)

probabilities = raw_probabilities

classes = model.named_steps[
    "logistic_regression"
].classes_
print("Classes:", classes)

raw_test_log_loss = log_loss(
    y_test,
    raw_probabilities,
    labels=classes,
)

calibrated_test_log_loss = log_loss(
    y_test,
    probabilities,
    labels=classes,
)

raw_brier = multiclass_brier_score(
    y_test,
    raw_probabilities,
    classes,
)

calibrated_brier = multiclass_brier_score(
    y_test,
    probabilities,
    classes,
)

print(
    "Raw Test Log Loss:",
    raw_test_log_loss,
)
print(
    "Calibrated Test Log Loss:",
    calibrated_test_log_loss,
)
print(
    "Raw Test Brier Score:",
    raw_brier,
)
print(
    "Calibrated Test Brier Score:",
    calibrated_brier,
)

accuracy = accuracy_score(
    y_test,
    predictions,
)

baseline_predictions = [
    "H"
] * len(y_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions,
)

print("Model Accuracy:", accuracy)
print(
    "Baseline Accuracy:",
    baseline_accuracy,
)
print(
    "Training matches:",
    len(x_train),
)
print(
    "Testing matches:",
    len(x_test),
)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        predictions,
    )
)

print("Confusion Matrix")
print(
    confusion_matrix(
        y_test,
        predictions,
    )
)
print(classes)


# ----------------------------------
# Results preview
# ----------------------------------

results = test_df.copy()
results["Prediction"] = predictions

results["AwayProbability"] = (
    probabilities[:, 0]
)
results["DrawProbability"] = (
    probabilities[:, 1]
)
results["HomeProbability"] = (
    probabilities[:, 2]
)

print("\nActual results:")
print(
    pd.Series(y_test).value_counts()
)

print("\nPredicted results:")
print(
    pd.Series(predictions).value_counts()
)

pd.set_option(
    "display.max_columns",
    None,
)
pd.set_option(
    "display.width",
    None,
)

print(
    results[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "Prediction",
            "HomeProbability",
            "DrawProbability",
            "AwayProbability",
        ]
    ].head(20)
)
