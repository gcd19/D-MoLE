#!/usr/bin/env python3
"""Reconstruct internal D-MoLE assets from public upstream sources.

This script intentionally keeps all reconstructed outputs local to the repo.
It currently supports the minimal VizWiz captioning lane needed to qualify the
single-GPU D-MoLE evidence path without relying on the blocked monolithic
Google Drive tarball.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable


VIZWIZ_CAPTION_ANNOTATIONS_URL = (
    "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip"
)
VIZWIZ_CAPTION_TRAIN_ZIP_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip"
VIZWIZ_CAPTION_PROMPT = "Provide a one-sentence caption for the provided image."


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    print(f"downloading {url} -> {destination}", flush=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(destination)


def extract_zip_member(zip_path: Path, member_prefix: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.startswith(member_prefix):
                continue
            target = output_dir / Path(member).name
            if target.exists() and target.stat().st_size > 0:
                continue
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def sanitize_caption(caption: str) -> str:
    return " ".join(caption.strip().split())


def download_many_image_urls(pairs: Iterable[tuple[str, Path]], workers: int) -> None:
    errors: list[str] = []

    def fetch(url: str, destination: Path) -> None:
        try:
            download_file(url, destination)
        except urllib.error.URLError as exc:
            errors.append(f"{url} -> {destination}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for url, destination in pairs:
            executor.submit(fetch, url, destination)

    if errors:
        raise RuntimeError(
            "failed to download one or more VizWiz caption images:\n"
            + "\n".join(errors[:10])
        )


def extract_selected_images_from_zip(
    zip_path: Path, file_names: Iterable[str], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(file_names)
    missing = requested.copy()
    with zipfile.ZipFile(zip_path) as archive:
        by_name = {Path(member).name: member for member in archive.namelist()}
        for file_name in requested:
            if file_name not in by_name:
                continue
            destination = output_dir / file_name
            if destination.exists() and destination.stat().st_size > 0:
                missing.discard(file_name)
                continue
            with archive.open(by_name[file_name]) as source, destination.open(
                "wb"
            ) as handle:
                shutil.copyfileobj(source, handle)
            missing.discard(file_name)
    if missing:
        raise RuntimeError(
            "missing expected VizWiz caption images in train.zip: "
            + ", ".join(sorted(list(missing))[:10])
        )


def reconstruct_vizwiz_caption(
    repo_root: Path, sample_size: int, seed: int, download_workers: int
) -> None:
    cache_root = Path(
        os.environ.get(
            "DMOLE_PUBLIC_CACHE_ROOT",
            str(Path.home() / ".cache" / "dmole-public-assets"),
        )
    )
    download_root = cache_root / "vizwiz-caption"
    annotations_zip = download_root / "annotations.zip"
    train_zip = download_root / "train.zip"
    annotations_dir = download_root / "annotations"
    train_annotations_path = annotations_dir / "train.json"
    data_root = repo_root / "data" / "vizwiz-caption"
    images_root = data_root / "images" / "train"
    manifest_path = data_root / "internal_minimal_train.jsonl"
    receipt_path = data_root / "public_source_receipt.json"
    meta_path = repo_root / "shell" / "dmole_internal" / "vizwiz_caption_minimal.json"

    download_file(VIZWIZ_CAPTION_ANNOTATIONS_URL, annotations_zip)
    if not train_annotations_path.exists():
        extract_zip_member(annotations_zip, "annotations/", annotations_dir)

    raw = json.loads(train_annotations_path.read_text(encoding="utf-8"))
    images_by_id = {item["id"]: item for item in raw["images"]}
    annotations = raw["annotations"]

    if sample_size <= 0 or sample_size >= len(annotations):
        selected_annotations = annotations
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(annotations)), sample_size))
        selected_annotations = [annotations[index] for index in indices]

    images_root.mkdir(parents=True, exist_ok=True)
    unique_downloads: dict[str, Path] = {}
    for annotation in selected_annotations:
        image_info = images_by_id[annotation["image_id"]]
        unique_downloads.setdefault(
            image_info["vizwiz_url"], images_root / image_info["file_name"]
        )

    try:
        download_many_image_urls(unique_downloads.items(), workers=download_workers)
    except RuntimeError as exc:
        print(
            "warning: direct VizWiz image URLs failed, falling back to the official train.zip "
            f"bundle ({exc})",
            flush=True,
        )
        download_file(VIZWIZ_CAPTION_TRAIN_ZIP_URL, train_zip)
        print(
            f"extracting {len(unique_downloads)} selected VizWiz images from {train_zip}",
            flush=True,
        )
        extract_selected_images_from_zip(
            train_zip,
            [path.name for path in unique_downloads.values()],
            images_root,
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for annotation in selected_annotations:
            image_info = images_by_id[annotation["image_id"]]
            record = {
                "id": f"vizwiz_caption_{annotation['id']}",
                "image": f"data/vizwiz-caption/images/train/{image_info['file_name']}",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{VIZWIZ_CAPTION_PROMPT}",
                    },
                    {
                        "from": "gpt",
                        "value": sanitize_caption(annotation["caption"]),
                    },
                ],
                "source": {
                    "dataset": "vizwiz-caption",
                    "annotation_id": annotation["id"],
                    "image_id": annotation["image_id"],
                    "annotation_zip_url": VIZWIZ_CAPTION_ANNOTATIONS_URL,
                    "image_url": image_info["vizwiz_url"],
                    "text_detected": annotation.get("text_detected"),
                    "is_precanned": annotation.get("is_precanned"),
                    "is_rejected": annotation.get("is_rejected"),
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_payload = {
        "vizwiz-caption": {
            "root": ".",
            "annotation": "data/vizwiz-caption/internal_minimal_train.jsonl",
            "data_augment": False,
            "repeat_time": 1,
            "length": len(selected_annotations),
        }
    }
    meta_path.write_text(
        json.dumps(meta_payload, indent=4, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    receipt_payload = {
        "task": "vizwiz_caption",
        "mode": "public_source_internal_reconstruction",
        "annotation_zip_url": VIZWIZ_CAPTION_ANNOTATIONS_URL,
        "train_zip_url": VIZWIZ_CAPTION_TRAIN_ZIP_URL,
        "cache_root": str(download_root),
        "annotation_json": display_path(train_annotations_path, repo_root),
        "output_manifest": display_path(manifest_path, repo_root),
        "output_meta": display_path(meta_path, repo_root),
        "sample_size": len(selected_annotations),
        "seed": seed,
        "image_count": len(unique_downloads),
        "prompt": VIZWIZ_CAPTION_PROMPT,
    }
    receipt_path.write_text(
        json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("vizwiz_caption",),
        default="vizwiz_caption",
        help="Public-source reconstruction task to execute.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo root for D-MoLE-Research.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=256,
        help="Number of public annotations to stage into the internal training manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260409,
        help="Deterministic seed for sampling public annotations.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=max(4, min(16, (os.cpu_count() or 4))),
        help="Parallel worker count for image downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.task == "vizwiz_caption":
        reconstruct_vizwiz_caption(
            repo_root,
            sample_size=args.sample_size,
            seed=args.seed,
            download_workers=args.download_workers,
        )


if __name__ == "__main__":
    main()
