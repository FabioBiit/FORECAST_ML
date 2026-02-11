from __future__ import annotations

from pathlib import Path
import polars as pl
from xgboost import XGBRegressor

from .train import build_features, FeatConfig, to_xy
from .utils import SplitConfig, metrics, time_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "power_hourly.parquet"


def train_and_evaluate_horizon(horizon: int, verbose: bool = True) -> dict[str, float]:
    """Train and evaluate model for a specific forecast horizon."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Testing horizon: {horizon}h")
        print(f"{'=' * 60}")

    # Load data
    df = pl.read_parquet(DATA_PATH).sort("dt")

    # Build features for this horizon
    fc = FeatConfig(horizon=horizon)
    df_feat = build_features(df, fc)

    feature_cols = [c for c in df_feat.columns if c not in ("dt", "y", "y_target")]

    # Split data
    train_df, val_df, test_df = time_split(df_feat, SplitConfig())

    X_train, y_train = to_xy(train_df, feature_cols)
    X_val, y_val = to_xy(val_df, feature_cols)
    X_test, y_test = to_xy(test_df, feature_cols)

    # Naive baseline
    yhat_val_naive = val_df["y"].to_numpy()
    yhat_test_naive = test_df["y"].to_numpy()

    naive_val = metrics(y_val, yhat_val_naive)
    naive_test = metrics(y_test, yhat_test_naive)

    # Train model
    model = XGBRegressor(
        n_estimators=500,  # Fewer trees for faster testing
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Predictions
    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)

    m_val = metrics(y_val, yhat_val)
    m_test = metrics(y_test, yhat_test)

    # Calculate improvements
    val_mae_improvement = ((naive_val["mae"] - m_val["mae"]) / naive_val["mae"]) * 100
    test_mae_improvement = ((naive_test["mae"] - m_test["mae"]) / naive_test["mae"]) * 100

    if verbose:
        print(f"\nNaive Baseline:")
        print(f"  Val:  MAE={naive_val['mae']:.4f} RMSE={naive_val['rmse']:.4f}")
        print(f"  Test: MAE={naive_test['mae']:.4f} RMSE={naive_test['rmse']:.4f}")

        print(f"\nXGBoost Model:")
        print(f"  Val:  MAE={m_val['mae']:.4f} RMSE={m_val['rmse']:.4f} (↓{val_mae_improvement:.1f}%)")
        print(f"  Test: MAE={m_test['mae']:.4f} RMSE={m_test['rmse']:.4f} (↓{test_mae_improvement:.1f}%)")

    return {
        "horizon": horizon,
        "naive_val_mae": naive_val["mae"],
        "naive_val_rmse": naive_val["rmse"],
        "model_val_mae": m_val["mae"],
        "model_val_rmse": m_val["rmse"],
        "naive_test_mae": naive_test["mae"],
        "naive_test_rmse": naive_test["rmse"],
        "model_test_mae": m_test["mae"],
        "model_test_rmse": m_test["rmse"],
        "val_improvement_pct": val_mae_improvement,
        "test_improvement_pct": test_mae_improvement,
    }


def main() -> None:
    """Test multiple forecast horizons."""
    print("=" * 80)
    print("TESTING MULTIPLE FORECAST HORIZONS")
    print("=" * 80)

    # Test different horizons
    horizons = [6, 12, 24, 48, 72, 168]  # 6h, 12h, 1d, 2d, 3d, 1w

    results = []
    for h in horizons:
        result = train_and_evaluate_horizon(h, verbose=True)
        results.append(result)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: PERFORMANCE ACROSS HORIZONS")
    print("=" * 80)
    print(f"\n{'Horizon':<10} {'Val MAE':<12} {'Test MAE':<12} {'Val Improv.':<15} {'Test Improv.':<15}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['horizon']:>3}h      "
            f"{r['model_val_mae']:<12.4f} "
            f"{r['model_test_mae']:<12.4f} "
            f"{r['val_improvement_pct']:<14.1f}% "
            f"{r['test_improvement_pct']:<14.1f}%"
        )

    # Find best horizon
    best_val = min(results, key=lambda x: x["model_val_mae"])
    best_test = min(results, key=lambda x: x["model_test_mae"])

    print("\n" + "=" * 80)
    print("BEST HORIZONS")
    print("=" * 80)
    print(f"Best on Validation: {best_val['horizon']}h (MAE={best_val['model_val_mae']:.4f})")
    print(f"Best on Test:       {best_test['horizon']}h (MAE={best_test['model_test_mae']:.4f})")
    print("=" * 80)

    # Save results to CSV
    import polars as pl_export
    results_df = pl_export.DataFrame(results)
    results_path = REPO_ROOT / "models" / "horizon_comparison.csv"
    results_df.write_csv(results_path)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
