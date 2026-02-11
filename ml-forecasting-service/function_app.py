from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime


def run_pipeline() -> None:
    """Run the complete ML forecasting pipeline in sequence."""
    start_time = datetime.now()
    print("=" * 80)
    print("ML FORECASTING PIPELINE - START")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # Step 1: Data Ingestion (NON NECESSARIO DA LANCIARE OGNI VOLTA)
        storico = int(input("Do you want to run data ingestion? This will fetch new data and overwrite existing dataset. (1 for yes, 0 for no): "))
        if storico == 1:
            print("\n[1/4] Running data ingestion...")
            print("-" * 80)
            from data_ingestion.fetch_data import main as fetch_main
            fetch_main()
            print("✓ Data ingestion completed successfully\n")
        else:
            print("\n[1/4] Skipping data ingestion. Using existing dataset.\n")

        # Step 2: Feature Engineering
        print("\n[2/4] Building dataset with features...")
        print("-" * 80)
        from features.build_dataset import main as build_main
        build_main()
        print("✓ Dataset built successfully\n")

        # Step 3: Baseline Model
        print("\n[3/4] Running baseline model...")
        print("-" * 80)
        from training.baseline import main as baseline_main
        baseline_main(horizon=24)
        print("✓ Baseline completed successfully\n")

        # Step 4: Model Training
        print("\n[4/4] Training XGBoost model...")
        print("-" * 80)
        from training.train import main as train_main
        train_main()
        print("✓ Model training completed successfully\n")

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("PIPELINE FAILED")
        print("=" * 80)
        print(f"Error: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
