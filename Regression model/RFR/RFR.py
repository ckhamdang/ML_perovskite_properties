#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load training data
data = pd.read_csv("../Dataset/ABX3_perovskite_cleaned.csv")

# Load out-of-sample data
out_sample = pd.read_csv("../Dataset/Out_of_sample_ABX3.csv")

feature_names = [
    "MA", "FA", "Cs", "EA",
    "Pb", "Sn",
    "Cl", "Br", "I"
]

target_names = [
    "Efficiency (%)",
    "Open circuit voltage (V)",
    "Short circuit current density (A/m²)",
    "Fill factor",
    "Band gap (eV)"
]

axis_settings = {
    "Efficiency (%)": {
        "xmin": -5, "xmax": 30, "ymin": -5, "ymax": 30,
        "xticks": [0, 5, 10, 15, 20, 25, 30],
        "yticks": [-5, 0, 5, 10, 15, 20, 25, 30]
    },
    "Open circuit voltage (V)": {
        "xmin": -0.3, "xmax": 1.8, "ymin": -0.3, "ymax": 1.8,
        "xticks": [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8],
        "yticks": [-0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
    },
    "Short circuit current density (A/m²)": {
        "xmin": -100, "xmax": 400, "ymin": -100, "ymax": 400,
        "xticks": [0, 100, 200, 300, 400],
        "yticks": [-100, 0, 100, 200, 300, 400]
    },
    "Fill factor": {
        "xmin": 0.2, "xmax": 1.0, "ymin": 0.2, "ymax": 1.0,
        "xticks": [0.4, 0.6, 0.8, 1.0],
        "yticks": [0.2, 0.4, 0.6, 0.8, 1.0]
    },
    "Band gap (eV)": {
        "xmin": 1.0, "xmax": 3.5, "ymin": 1.0, "ymax": 3.5,
        "xticks": [1.5, 2.0, 2.5, 3.0, 3.5],
        "yticks": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    }
}

text_positions = {
    "Efficiency (%)": (0.65, 0.05),
    "Open circuit voltage (V)": (0.65, 0.05),
    "Short circuit current density (A/m²)": (0.65, 0.05),
    "Fill factor": (0.65, 0.05),
    "Band gap (eV)": (0.65, 0.05)
}

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 10, 20],
    "min_samples_split": [2, 5, 10, 15]
}

model_results = {}

# Train models
for target in target_names:

    print("\n" + "=" * 60)
    print("Training target:", target)
    print("=" * 60)

    df = data.dropna(subset=feature_names + [target])

    X = df[feature_names]
    y = df[target]

    X_fl = np.array(X, dtype=float)
    y_fl = np.array(y, dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_fl, y_fl, test_size=0.2, random_state=42
    )

    rfreg = RandomForestRegressor(random_state=42, n_jobs=1)

    rfr_opt = GridSearchCV(
        estimator=rfreg,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=1
    )

    rfr_opt.fit(X_train, y_train)

    best_model = rfr_opt.best_estimator_

    print("Best estimator:", best_model)
    
    Pred_train = best_model.predict(X_train)
    Pred_test = best_model.predict(X_test)

    r2_train = r2_score(y_train, Pred_train)
    r2_test = r2_score(y_test, Pred_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, Pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, Pred_test))

    print("Train R² =", r2_train)
    print("Test R²  =", r2_test)
    print("Train RMSE =", rmse_train)
    print("Test RMSE  =", rmse_test)

    model_results[target] = {
        "model": best_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "Pred_train": Pred_train,
        "Pred_test": Pred_test,
        "r2_train": r2_train,
        "r2_test": r2_test
    }

    # Parity plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        y_train, Pred_train,
        color="#FA7070",
        label="Train",
        s=150,
        edgecolor="k",
        marker="o"
    )

    ax.scatter(
        y_test, Pred_test,
        color="#615EFC",
        label="Test",
        s=150,
        edgecolor="k",
        marker="o"
    )

    settings = axis_settings[target]

    ax.plot(
        [settings["xmin"], settings["xmax"]],
        [settings["xmin"], settings["xmax"]],
        c="k",
        ls="-"
    )

    ax.set_xlim([settings["xmin"], settings["xmax"]])
    ax.set_ylim([settings["ymin"], settings["ymax"]])
    ax.set_xticks(settings["xticks"])
    ax.set_yticks(settings["yticks"])

    ax.set_xlabel("Experimental", fontsize=24)
    ax.set_ylabel("ML Prediction", fontsize=24)
    ax.set_title(target, fontsize=26, pad=10)

    x_text, y_text = text_positions[target]

    ax.text(
        x_text,
        y_text,
        rf"Train $R^2$ = {r2_train:.2f}" "\n"
        rf"Test $R^2$ = {r2_test:.2f}",
        transform=ax.transAxes,
        fontsize=18
    )

    plt.tick_params(axis="y", width=2, length=8, labelsize=24)
    plt.tick_params(axis="x", width=2, length=8, labelsize=24)

    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    ax.legend(fontsize=20)

    plt.tight_layout()

    filename = target.replace(" ", "_")
    filename = filename.replace("(", "").replace(")", "")
    filename = filename.replace("/", "_").replace("²", "2")

    plt.savefig(f"RFR_{filename}.png", dpi=450)
    plt.show()

# SHAP analysis after all models are trained
writer = pd.ExcelWriter("RFR_SHAP_values_all_targets.xlsx", engine="openpyxl")
summary_rows = []

for target in target_names:

    result = model_results[target]
    best_model = result["model"]

    X_train = result["X_train"]
    X_test = result["X_test"]

    y_train = result["y_train"]
    y_test = result["y_test"]

    Pred_train = result["Pred_train"]
    Pred_test = result["Pred_test"]

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    explainer = shap.TreeExplainer(best_model)

    shap_train = explainer.shap_values(X_train_df)
    shap_test = explainer.shap_values(X_test_df)

    shap_train_df = pd.DataFrame(
        shap_train,
        columns=[f"SHAP_{f}" for f in feature_names]
    )

    shap_test_df = pd.DataFrame(
        shap_test,
        columns=[f"SHAP_{f}" for f in feature_names]
    )

    train_result = pd.concat(
        [
            X_train_df.reset_index(drop=True),
            pd.Series(y_train, name=f"Actual_{target}"),
            pd.Series(Pred_train, name=f"Predicted_{target}"),
            shap_train_df.reset_index(drop=True)
        ],
        axis=1
    )

    train_result["Dataset"] = "Train"

    test_result = pd.concat(
        [
            X_test_df.reset_index(drop=True),
            pd.Series(y_test, name=f"Actual_{target}"),
            pd.Series(Pred_test, name=f"Predicted_{target}"),
            shap_test_df.reset_index(drop=True)
        ],
        axis=1
    )

    test_result["Dataset"] = "Test"

    shap_output = pd.concat(
        [train_result, test_result],
        ignore_index=True
    )

    cols = (
        ["Dataset"] +
        feature_names +
        [f"Actual_{target}", f"Predicted_{target}"] +
        [f"SHAP_{f}" for f in feature_names]
    )

    shap_output = shap_output[cols]

    sheet_name = (
        target.replace("Efficiency (%)", "Efficiency")
              .replace("Open circuit voltage (V)", "Voc")
              .replace("Short circuit current density (A/m²)", "Jsc")
              .replace("Fill factor", "FF")
              .replace("Band gap (eV)", "Eg")
    )

    shap_output.to_excel(writer, sheet_name=sheet_name, index=False)

    mean_abs_shap = np.abs(shap_train).mean(axis=0)

    for feature, value in zip(feature_names, mean_abs_shap):
        summary_rows.append({
            "Target": target,
            "Feature": feature,
            "Mean_abs_SHAP": value
        })

    shap.summary_plot(
        shap_train,
        X_train_df,
        feature_names=feature_names,
        show=False
    )

    safe_target = target.replace(" ", "_")
    safe_target = safe_target.replace("(", "").replace(")", "")
    safe_target = safe_target.replace("/", "_").replace("²", "2")

    plt.tight_layout()
    plt.savefig(f"SHAP_RFR_{safe_target}.png", dpi=450)
    plt.show()

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel(writer, sheet_name="Mean_abs_SHAP_summary", index=False)

writer.close()

# Out-of-sample prediction
out_sample_clean = out_sample.dropna(subset=feature_names).copy()
X_out = out_sample_clean[feature_names]
X_out_fl = np.array(X_out, dtype=float)

for target in target_names:

    best_model = model_results[target]["model"]

    Pred_out = best_model.predict(X_out_fl)

    out_sample.loc[
        out_sample_clean.index,
        f"Predicted {target}"
    ] = Pred_out

out_sample.to_excel(
    "Out_of_sample_ABX3_RFR_predictions.xlsx",
    index=False
)

print()
print("Saved SHAP values to RFR_SHAP_values_all_targets.xlsx")
print("Saved out-of-sample predictions to Out_of_sample_ABX3_RFR_predictions.xlsx")