"""Scan locally built/pulled images, preserving machine-readable evidence."""

import argparse
import json
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCANNER = (
    "aquasec/trivy:0.74.0@sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969"
)


def scan(images, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    with tempfile.TemporaryDirectory(prefix="fraud-scan-") as folder:
        archive = Path(folder) / "image.tar"
        for number, image in enumerate(images):
            subprocess.run(["docker", "save", "-o", str(archive), image], check=True)
            report = output / f"image-{number}.json"
            with report.open("w") as destination:
                subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "-v",
                        f"{archive}:/scan/image.tar:ro",
                        "-v",
                        "fraud-trivy-cache:/root/.cache/trivy",
                        SCANNER,
                        "image",
                        "--quiet",
                        "--input",
                        "/scan/image.tar",
                        "--scanners",
                        "vuln",
                        "--severity",
                        "HIGH,CRITICAL",
                        "--format",
                        "json",
                    ],
                    stdout=destination,
                    check=True,
                )
            data = json.loads(report.read_text())
            vulnerabilities = [
                v for result in data.get("Results", []) for v in result.get("Vulnerabilities", [])
            ]
            summary = {
                "image": image,
                "image_id": data.get("Metadata", {}).get("ImageID"),
                "scanned_at": datetime.now(timezone.utc).isoformat(),
                "scanner": SCANNER,
                "high_critical_count": len(vulnerabilities),
                "status_counts": dict(Counter(v.get("Status", "unknown") for v in vulnerabilities)),
                "report": report.name,
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return int(any(x["high_critical_count"] for x in summaries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/security"))
    args = parser.parse_args()
    raise SystemExit(scan(args.image, args.output))


if __name__ == "__main__":
    main()
