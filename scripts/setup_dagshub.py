"""Configure DVC remote and MLflow tracking for DagsHub from environment variables.

Prerequisites:
    1. Copy .env.example to .env
    2. Fill in DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, and DAGSHUB_TOKEN
    3. Run: python scripts/setup_dagshub.py

This script is idempotent — running it multiple times is safe.
"""
import os
import subprocess
import sys
from pathlib import Path


def load_env():
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def run(cmd: list[str], check: bool = True):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=check)


def main():
    load_env()

    owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if not all([owner, repo, token]):
        print("ERROR: Set DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, and DAGSHUB_TOKEN in .env")
        sys.exit(1)

    remote_url = f"https://dagshub.com/{owner}/{repo}.dvc"
    mlflow_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"

    # Configure DVC remote (DagsHub HTTPS remote)
    run(["dvc", "remote", "add", "-f", "origin", remote_url], check=False)
    run(["dvc", "remote", "default", "origin"])
    run(["dvc", "remote", "modify", "origin", "--local", "auth", "basic"])
    run(["dvc", "remote", "modify", "origin", "--local", "user", owner])
    run(["dvc", "remote", "modify", "origin", "--local", "password", token])

    print(f"\nDVC remote configured: {remote_url}")
    print(f"MLflow tracking URI:   {mlflow_uri}")
    print("\nNext steps:")
    print("  dvc push")
    print("  git add .dvc/config .dvc/config.local")
    print("  git commit -m 'Configure DagsHub DVC remote'")


if __name__ == "__main__":
    main()
