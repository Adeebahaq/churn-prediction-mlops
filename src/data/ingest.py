"""Download the Telco Customer Churn dataset and save it locally."""
import argparse
import logging
import requests
from pathlib import Path

from src.utils import load_params, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: str) -> None:
    ensure_dir(dest_path)
    logger.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)
    logger.info("Saved raw dataset to %s", dest_path)


def ingest(raw_path: str, source_url: str) -> str:
    path = Path(raw_path)
    if path.exists():
        logger.info("Raw file already exists at %s; skipping download.", raw_path)
        return raw_path
    download_file(source_url, raw_path)
    return raw_path


def main():
    parser = argparse.ArgumentParser(description="Ingest Telco Customer Churn data")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    ingest(params["data"]["raw_path"], params["data"]["source_url"])


if __name__ == "__main__":
    main()
