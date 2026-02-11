from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance

from .utils import SplitConfig, metrics, time_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "power_hourly.parquet"
MODEL_PATH = REPO_ROOT / "models" / "xgb_power_h24.joblib"
PLOTS_DIR = REPO_ROOT / "plots"


def plot_predictions_over_time(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot actual vs predicted values over time."""
    plt.figure(figsize=(15, 6))

    # Plot first 500 points for readability
    n_points = min(500, len(dates))
    plt.plot(dates[:n_points], y_true[:n_points], label="Actual", alpha=0.7, linewidth=1.5)
    plt.plot(dates[:n_points], y_pred[:n_points], label="Predicted", alpha=0.7, linewidth=1.5)

    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Power Consumption (kW)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot: {save_path}")


def plot_residuals_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Analyze residuals with multiple plots."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residuals over time
    axes[0, 0].plot(residuals, alpha=0.6, linewidth=0.8)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Sample Index")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Histogram of residuals
    axes[0, 1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[0, 1].set_xlabel("Residual Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(f"Residuals Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Predicted vs Residuals (check heteroscedasticity)
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1, 0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1, 0].set_xlabel("Predicted Values")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("Residuals vs Predicted")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Q-Q plot (check normality)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot (Normality Check)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot: {save_path}")


def plot_feature_importance(model, feature_cols: list[str], save_path: Path) -> None:
    """Plot feature importance from XGBoost model."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get feature importance
    importance_dict = model.get_booster().get_score(importance_type="weight")

    # Map feature names
    feature_map = {f"f{i}": name for i, name in enumerate(feature_cols)}
    importance_named = {feature_map.get(k, k): v for k, v in importance_dict.items()}

    # Sort by importance
    sorted_importance = sorted(importance_named.items(), key=lambda x: x[1], reverse=True)
    features, scores = zip(*sorted_importance[:15])  # Top 15

    # Plot horizontal bar chart
    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, alpha=0.8, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Weight)", fontsize=12)
    ax.set_title("Top 15 Most Important Features", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot: {save_path}")


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: Path) -> None:
    """Scatter plot of actual vs predicted values."""
    plt.figure(figsize=(8, 8))

    plt.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)

    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"{title}\nR² = {r2:.4f}", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot: {save_path}")


def plot_error_distribution_by_hour(
    dates: pl.Series,
    residuals: np.ndarray,
    save_path: Path,
) -> None:
    """Analyze prediction errors by hour of day."""
    df_errors = pl.DataFrame({
        "dt": dates,
        "residual": residuals,
        "abs_error": np.abs(residuals),
    })

    df_errors = df_errors.with_columns(pl.col("dt").dt.hour().alias("hour"))

    hourly_stats = df_errors.group_by("hour").agg([
        pl.col("abs_error").mean().alias("mae"),
        pl.col("residual").mean().alias("mean_residual"),
        pl.col("residual").std().alias("std_residual"),
    ]).sort("hour")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAE by hour
    axes[0].bar(hourly_stats["hour"], hourly_stats["mae"], alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Hour of Day", fontsize=12)
    axes[0].set_ylabel("Mean Absolute Error", fontsize=12)
    axes[0].set_title("Prediction Error by Hour", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_xticks(range(0, 24, 2))

    # Mean residual by hour (bias check)
    axes[1].bar(hourly_stats["hour"], hourly_stats["mean_residual"], alpha=0.7, color="coral")
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Hour of Day", fontsize=12)
    axes[1].set_ylabel("Mean Residual (Bias)", fontsize=12)
    axes[1].set_title("Prediction Bias by Hour", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot: {save_path}")


def main() -> None:
    """Run complete model analysis."""
    print("=" * 80)
    print("MODEL ANALYSIS - START")
    print("=" * 80)

    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and data
    print("\n[1/6] Loading model and data...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run training/train.py first.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    horizon = bundle["horizon"]

    print(f"  Model: XGBoost with horizon={horizon}h")
    print(f"  Features: {len(feature_cols)} features")

    # Load and prepare data (same as train.py)
    df = pl.read_parquet(DATA_PATH).sort("dt")

    # Rebuild features (import from train.py logic)
    from .train import build_features, FeatConfig, to_xy

    fc = FeatConfig(horizon=horizon)
    df_feat = build_features(df, fc)

    train_df, val_df, test_df = time_split(df_feat, SplitConfig())

    # Get predictions
    X_val, y_val = to_xy(val_df, feature_cols)
    X_test, y_test = to_xy(test_df, feature_cols)

    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)

    val_metrics = metrics(y_val, yhat_val)
    test_metrics = metrics(y_test, yhat_test)

    print(f"  Val:  MAE={val_metrics['mae']:.4f} RMSE={val_metrics['rmse']:.4f}")
    print(f"  Test: MAE={test_metrics['mae']:.4f} RMSE={test_metrics['rmse']:.4f}")

    # Analysis 1: Predictions over time
    print("\n[2/6] Generating predictions over time plots...")
    plot_predictions_over_time(
        val_df["dt"].to_numpy(),
        y_val,
        yhat_val,
        "Validation Set: Actual vs Predicted",
        PLOTS_DIR / "val_predictions_time.png",
    )
    plot_predictions_over_time(
        test_df["dt"].to_numpy(),
        y_test,
        yhat_test,
        "Test Set: Actual vs Predicted",
        PLOTS_DIR / "test_predictions_time.png",
    )

    # Analysis 2: Feature importance
    print("\n[3/6] Analyzing feature importance...")
    plot_feature_importance(model, feature_cols, PLOTS_DIR / "feature_importance.png")

    # Analysis 3: Residuals analysis
    print("\n[4/6] Analyzing residuals...")
    plot_residuals_analysis(
        y_val,
        yhat_val,
        "Validation Set: Residuals Analysis",
        PLOTS_DIR / "val_residuals.png",
    )
    plot_residuals_analysis(
        y_test,
        yhat_test,
        "Test Set: Residuals Analysis",
        PLOTS_DIR / "test_residuals.png",
    )

    # Analysis 4: Actual vs Predicted scatter
    print("\n[5/6] Generating actual vs predicted plots...")
    plot_actual_vs_predicted(
        y_val,
        yhat_val,
        "Validation Set: Actual vs Predicted",
        PLOTS_DIR / "val_scatter.png",
    )
    plot_actual_vs_predicted(
        y_test,
        yhat_test,
        "Test Set: Actual vs Predicted",
        PLOTS_DIR / "test_scatter.png",
    )

    # Analysis 5: Error by hour of day
    print("\n[6/6] Analyzing errors by hour...")
    plot_error_distribution_by_hour(
        test_df["dt"],
        y_test - yhat_test,
        PLOTS_DIR / "error_by_hour.png",
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"\nAll plots saved in: {PLOTS_DIR}")
    print("\nGenerated plots:")
    print("  1. val_predictions_time.png  - Predictions over time (validation)")
    print("  2. test_predictions_time.png - Predictions over time (test)")
    print("  3. feature_importance.png    - Top 15 most important features")
    print("  4. val_residuals.png         - Residuals analysis (validation)")
    print("  5. test_residuals.png        - Residuals analysis (test)")
    print("  6. val_scatter.png           - Actual vs predicted (validation)")
    print("  7. test_scatter.png          - Actual vs predicted (test)")
    print("  8. error_by_hour.png         - Errors by hour of day")
    print("=" * 80)


if __name__ == "__main__":
    main()
