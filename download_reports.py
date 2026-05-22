from pathlib import Path
import json

from huggingface_hub import snapshot_download

repo_id = "bobotsalos/news_reports"  # change to your dataset repo id

# Download only the reports folder contents into the current project directory.
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=".",
    allow_patterns=["reports/**/*.json", "reports/*.json"],
)

# Now iterate the local files.
reports_dir = Path("reports")
for json_path in reports_dir.rglob("*.json"):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(json_path, type(data))