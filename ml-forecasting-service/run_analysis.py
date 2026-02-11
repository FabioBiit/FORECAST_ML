from __future__ import annotations

import sys
from datetime import datetime


def main() -> None:
    """Run model analysis and visualizations."""
    start_time = datetime.now()
    print("=" * 80)
    print("MODEL ANALYSIS & VISUALIZATION")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # Run model analysis
        print("\n[STEP 1] Analyzing trained model and generating visualizations...")
        print("-" * 80)
        from training.analyze_model import main as analyze_main
        analyze_main()

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration}")
        print("\nCheck the 'plots/' folder for all generated visualizations.")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("ANALYSIS FAILED")
        print("=" * 80)
        print(f"Error: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
