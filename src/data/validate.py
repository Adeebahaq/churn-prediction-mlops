"""Validate raw data schema, completeness, and basic distributions."""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.utils import load_params, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate(raw_path: str, validation_cfg: dict, data_cfg: dict, target_column: str = "Churn") -> dict:
    df = pd.read_csv(raw_path)
    report = {
        "file": raw_path,
        "rows": len(df),
        "columns": list(df.columns),
        "n_columns": len(df.columns),
        "missing_pct": round(df.isnull().mean().mean() * 100, 4),
        "checks": {},
    }

    checks = report["checks"]

    # Schema checks
    checks["expected_rows"] = len(df) == data_cfg["expected_rows"]
    checks["expected_columns"] = len(df.columns) == data_cfg["expected_columns"]
    checks["has_target"] = target_column in df.columns

    # Missing-value checks
    checks["missing_pct_ok"] = report["missing_pct"] <= validation_cfg["max_missing_pct"]

    total_charges_missing = df["TotalCharges"].replace(" ", pd.NA).isnull().sum()
    checks["total_charges_missing_ok"] = int(total_charges_missing) <= validation_cfg["max_null_total_charges"]
    report["total_charges_missing"] = int(total_charges_missing)

    # Label distribution check
    churn_col = target_column
    churn_rate = df[churn_col].str.lower().eq("yes").mean()
    checks["churn_rate_ok"] = validation_cfg["churn_rate_min"] <= churn_rate <= validation_cfg["churn_rate_max"]
    report["churn_rate"] = round(churn_rate, 4)

    # Range checks on numeric columns
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col == "TotalCharges":
            df[col] = pd.to_numeric(df[col].replace(" ", pd.NA), errors="coerce")
        if col in df.columns:
            checks[f"{col}_nonnegative"] = (df[col].dropna() >= 0).all()

    checks["no_duplicate_customer_ids"] = not df["customerID"].duplicated().any()

    report["passed"] = bool(all(checks.values()))
    for key in checks:
        checks[key] = bool(checks[key])
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate raw Telco churn data")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output", default="metrics/validation_report.json")
    args = parser.parse_args()

    params = load_params(args.params)
    report = validate(
        params["data"]["raw_path"],
        params["validation"],
        params["data"],
        target_column=params["base"]["target_column"],
    )

    ensure_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation report saved to %s", args.output)
    logger.info("Validation passed: %s", report["passed"])

    if not report["passed"]:
        failed = [k for k, v in report["checks"].items() if not v]
        logger.error("Failed checks: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
