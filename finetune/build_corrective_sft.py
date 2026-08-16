import json
from pathlib import Path
from collections import Counter


# ============================================================
# SENTINELAI - BUILD CORRECTIVE SFT DATASET
#
# Combines:
#
#   1. Original 108 SFT records
#   2. 31 HUMAN-CORRECTED MODEL ERROR windows
#
# IMPORTANT:
# - Does NOT modify the original dataset.
# - Does NOT modify the targeted error dataset.
# - Does NOT use the remaining human test windows.
# - Does NOT train the model.
#
# Output:
#   C:\SentinelAI_Qwen\dataset\sft\corrective_sft_train.json
# ============================================================


PROJECT_ROOT = Path(r"C:\SentinelAI_Qwen")

ORIGINAL_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "sft"
    / "sft_train.json"
)

ERROR_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "sft"
    / "targeted_error_sft.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "sft"
    / "corrective_sft_train.json"
)


VALID_LABELS = {
    "Normal",
    "Fire",
    "Fight",
    "Road Accident",
}


def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain a JSON list."
        )

    return data


def validate_original(data):

    for i, item in enumerate(data):

        if not isinstance(item, dict):
            raise ValueError(
                f"Original record {i} is not an object."
            )

        required = [
            "video",
            "messages",
            "label",
        ]

        missing = [
            key
            for key in required
            if key not in item
        ]

        if missing:
            raise ValueError(
                f"Original record {i} missing: {missing}"
            )

        if item["label"] not in VALID_LABELS:
            raise ValueError(
                f"Invalid original label at "
                f"record {i}: {item['label']}"
            )


def validate_error(data):

    for i, item in enumerate(data):

        required = [
            "video",
            "start",
            "end",
            "label",
            "messages",
        ]

        missing = [
            key
            for key in required
            if key not in item
        ]

        if missing:
            raise ValueError(
                f"Error record {i} missing: {missing}"
            )

        if item["label"] not in VALID_LABELS:
            raise ValueError(
                f"Invalid error label at "
                f"record {i}: {item['label']}"
            )

        metadata = item.get(
            "metadata",
            {}
        )

        if not metadata.get(
            "targeted_error",
            False
        ):
            raise ValueError(
                f"Error record {i} is not marked "
                f"targeted_error=true."
            )


def make_unique_id(
    original_id,
    index,
):

    return (
        f"corrective_{index:04d}_"
        f"{original_id}"
    )


def main():

    print()
    print("=" * 70)
    print("SENTINELAI - BUILD CORRECTIVE SFT DATASET")
    print("=" * 70)

    print()
    print("Original SFT:")
    print(ORIGINAL_FILE)

    print()
    print("Human-corrected errors:")
    print(ERROR_FILE)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Load.
    # --------------------------------------------------------

    original = load_json(
        ORIGINAL_FILE
    )

    errors = load_json(
        ERROR_FILE
    )

    print()
    print("=" * 70)
    print("SOURCE DATA")
    print("=" * 70)

    print(
        f"Original records: "
        f"{len(original)}"
    )

    print(
        f"Human error corrections: "
        f"{len(errors)}"
    )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    validate_original(
        original
    )

    validate_error(
        errors
    )

    # --------------------------------------------------------
    # Prevent accidental duplicate records.
    #
    # We do not remove legitimate original records.
    # We only ensure the exact same error record isn't added
    # twice.
    # --------------------------------------------------------

    seen_error_keys = set()

    unique_errors = []

    for item in errors:

        key = (
            str(
                Path(
                    item["video"]
                )
            ).lower(),
            round(
                float(item["start"]),
                3
            ),
            round(
                float(item["end"]),
                3
            ),
            item["label"],
        )

        if key in seen_error_keys:
            continue

        seen_error_keys.add(
            key
        )

        unique_errors.append(
            item
        )

    # --------------------------------------------------------
    # Build final dataset.
    # --------------------------------------------------------

    final_data = []

    # Preserve original records exactly.
    for item in original:
        final_data.append(item)

    # Add human-corrected records.
    start_index = len(final_data) + 1

    for offset, item in enumerate(
        unique_errors
    ):

        new_item = dict(item)

        old_id = item.get(
            "id",
            f"error_{offset + 1:04d}"
        )

        new_item["id"] = make_unique_id(
            old_id,
            start_index + offset,
        )

        # Explicitly preserve the fact that this record
        # came from human correction.
        metadata = dict(
            item.get(
                "metadata",
                {}
            )
        )

        metadata[
            "training_source"
        ] = "human_corrected_test_error"

        metadata[
            "human_verified"
        ] = True

        metadata[
            "added_to_corrective_training"
        ] = True

        new_item[
            "metadata"
        ] = metadata

        final_data.append(
            new_item
        )

    # --------------------------------------------------------
    # Final validation.
    # --------------------------------------------------------

    if len(final_data) != (
        len(original)
        + len(unique_errors)
    ):
        raise RuntimeError(
            "Final dataset size check failed."
        )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Distribution.
    # --------------------------------------------------------

    counts = Counter(
        item["label"]
        for item in final_data
    )

    original_counts = Counter(
        item["label"]
        for item in original
    )

    error_counts = Counter(
        item["label"]
        for item in unique_errors
    )

    print()
    print("=" * 70)
    print("FINAL CORRECTIVE DATASET")
    print("=" * 70)

    print(
        f"Original records: "
        f"{len(original)}"
    )

    print(
        f"Unique human corrections added: "
        f"{len(unique_errors)}"
    )

    print(
        f"Final records: "
        f"{len(final_data)}"
    )

    print()
    print(
        f"{'Class':<20}"
        f"{'Original':>10}"
        f"{'Corrections':>13}"
        f"{'Final':>10}"
    )

    for label in [
        "Normal",
        "Fire",
        "Fight",
        "Road Accident",
    ]:

        print(
            f"{label:<20}"
            f"{original_counts[label]:>10}"
            f"{error_counts[label]:>13}"
            f"{counts[label]:>10}"
        )

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        "Original sft_train.json was NOT modified."
    )

    print(
        "targeted_error_sft.json was NOT modified."
    )

    print(
        "No other human test windows were added."
    )

    print(
        "NO TRAINING WAS PERFORMED."
    )

    print()
    print(
        "WARNING: The 31 corrections originated from "
        "the previous human evaluation set."
    )

    print(
        "Therefore, that 265-window set must NOT be "
        "used as an untouched test set for the new model."
    )

    print(
        "A fresh held-out evaluation set will be required "
        "after corrective training."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()