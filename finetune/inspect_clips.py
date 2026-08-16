import json
import random
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


# ============================================================
# SENTINELAI - TEMPORAL CLIP INSPECTOR
# ============================================================

MANIFEST = Path(
    r"C:\SentinelAI_Qwen\dataset\clips\train.jsonl"
)

OUTPUT_DIR = Path(
    r"C:\SentinelAI_Qwen\dataset\clip_inspection"
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SAMPLES_PER_CLASS = 5

CLASSES = [
    "Fire",
    "Fight",
    "Road Accident",
]


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest():

    records = []

    with open(
        MANIFEST,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if line:

                records.append(
                    json.loads(line)
                )

    return records


# ============================================================
# EXTRACT CLIP
# ============================================================

def extract_clip(
    record,
    output_path,
):

    video = record[
        "video_path"
    ]

    start = record[
        "start"
    ]

    duration = (
        record["end"]
        - record["start"]
    )

    command = [

        FFMPEG,

        "-y",

        "-ss",
        str(start),

        "-i",
        video,

        "-t",
        str(duration),

        "-an",

        "-c:v",
        "libx264",

        "-preset",
        "ultrafast",

        "-crf",
        "23",

        "-pix_fmt",
        "yuv420p",

        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SENTINELAI - TEMPORAL CLIP INSPECTION"
    )
    print("=" * 70)

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST}"
        )

    records = load_manifest()

    rng = random.Random(42)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Manifest records: "
        f"{len(records)}"
    )

    print(
        f"Samples per class: "
        f"{SAMPLES_PER_CLASS}"
    )

    print()

    for label in CLASSES:

        candidates = [

            r

            for r in records

            if r["label"] == label
        ]

        selected = rng.sample(
            candidates,
            min(
                SAMPLES_PER_CLASS,
                len(candidates)
            )
        )

        class_dir = (
            OUTPUT_DIR
            / label.replace(
                " ",
                "_"
            )
        )

        class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"{label}: "
            f"{len(selected)} samples"
        )

        for index, record in enumerate(
            selected,
            start=1,
        ):

            filename = (
                f"{index:02d}_"
                f"{record['clip_id']}.mp4"
            )

            output_path = (
                class_dir
                / filename
            )

            print(
                f"  [{index}] "
                f"{record['video_path']}"
            )

            print(
                f"      "
                f"{record['start']:.2f}s → "
                f"{record['end']:.2f}s"
            )

            extract_clip(
                record,
                output_path
            )

    print()
    print("=" * 70)
    print(
        "INSPECTION CLIPS CREATED"
    )
    print("=" * 70)

    print(
        f"Location:\n{OUTPUT_DIR}"
    )

    print()
    print(
        "Open the folders and watch the "
        "clips manually."
    )

    print(
        "Do NOT start LoRA training yet."
    )


if __name__ == "__main__":

    main()