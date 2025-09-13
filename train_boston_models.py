import os
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
import joblib


# 📊 Train & Evaluate
def train_and_evaluate(df, test_size=0.2, seed=42):
    # Detect target column
    target_col = next((c for c in df.columns if c.strip().lower() == "medv"), df.columns[-1])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Preprocessing
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        [("num", numeric_transformer, numeric_features),
         ("cat", categorical_transformer, categorical_features)]
    )

    # Models
    models = {
        "LinearRegression": Pipeline([("prep", preprocessor), ("model", LinearRegression())]),
        "RidgeCV": Pipeline([
            ("prep", preprocessor),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 15), cv=3))
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1))
        ]),
    }

    results = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        row = {"Model": name, "Test_RMSE": round(rmse, 4), "Test_R2": round(r2, 4)}
        if name == "RidgeCV":
            row["Chosen_alpha"] = float(pipe.named_steps["model"].alpha_)
        results.append(row)

    results_df = pd.DataFrame(results).sort_values(by="Test_RMSE").reset_index(drop=True)

    # Best model
    best_name = results_df.iloc[0]["Model"]
    best_pipe = models[best_name]

    # Save artifacts
    joblib.dump(best_pipe, "boston_best_model.pkl")
    with open("feature_columns.json", "w") as f:
        json.dump({
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "all_features": X.columns.tolist(),
            "target": target_col,
        }, f, indent=2)

    with open("model_card.md", "w") as f:
        f.write(f"# Boston Housing Model Card\n\n")
        f.write(f"- **Best Model**: {best_name}\n")
        f.write(f"- **Test RMSE**: {results_df.iloc[0]['Test_RMSE']}\n")
        f.write(f"- **Test R²**: {results_df.iloc[0]['Test_R2']}\n")

    # Diagnostics
    y_pred_best = best_pipe.predict(X_test)
    residuals = y_test - y_pred_best

    return results_df, best_name, y_test, y_pred_best, residuals


# 🚀 Streamlit App
def main():
    st.title("🏠 Boston Housing Price Prediction")
    st.write("Upload a dataset and train models with diagnostics.")

    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file).dropna().reset_index(drop=True)
        st.write("### Preview of Data", df.head())

        if st.button("Train Models"):
            results_df, best_name, y_test, y_pred_best, residuals = train_and_evaluate(df)

            # Show results
            st.subheader("📊 Model Comparison")
            st.dataframe(results_df)

            st.success(f"✅ Best Model: **{best_name}**")

            # Residual Plot
            st.subheader(f"Residual Plot — {best_name}")
            fig, ax = plt.subplots()
            ax.scatter(y_pred_best, residuals, alpha=0.7)
            ax.axhline(0, linestyle="--", color="red")
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals (y_true - y_pred)")
            st.pyplot(fig)

            # Predicted vs Actual
            st.subheader(f"Predicted vs Actual — {best_name}")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred_best, alpha=0.7)
            lo = min(y_test.min(), y_pred_best.min())
            hi = max(y_test.max(), y_pred_best.max())
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="red")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            st.pyplot(fig)

            st.info("Artifacts saved: `boston_best_model.pkl`, `feature_columns.json`, `model_card.md`")


if __name__ == "__main__":
    main()
    import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# --- After training your model ---
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Residual Plot
plt.figure(figsize=(6, 4))
sns.residplot(x=y_test_pred, y=(y_test - y_test_pred), lowess=True, 
              line_kws={"color": "red", "lw": 2})
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("residual_plot.png")
plt.close()

# Predicted vs Actual
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--", lw=2)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Predicted vs Actual")
plt.savefig("pred_vs_actual.png")
plt.close()
