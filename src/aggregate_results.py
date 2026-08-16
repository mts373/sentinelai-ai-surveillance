from pathlib import Path
import json
import sys


# ============================================================
# CONFIGURATION
# ============================================================

INCIDENT_CLASSES = {
    "Fire",
    "Fight",
    "Road Accident",
}

NORMAL_CLASS = "Normal"

# Windows closer than this are considered part of
# the same temporal incident episode.
TEMPORAL_GAP = 5.0


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# EXTRACT INCIDENT WINDOWS
# ============================================================

def get_incident_windows(windows):

    incidents = []

    for window in windows:

        classification = (
            window["classification"]
        )

        if classification not in INCIDENT_CLASSES:
            continue

        incidents.append(
            {
                "window_index":
                    window["window_index"],

                "start":
                    float(
                        window["start_seconds"]
                    ),

                "end":
                    float(
                        window["end_seconds"]
                    ),

                "classification":
                    classification,

                "fire":
                    window["fire"],

                "fight":
                    window["fight"],

                "road_accident":
                    window["road_accident"],
            }
        )

    return sorted(
        incidents,
        key=lambda x: x["start"]
    )


# ============================================================
# GROUP TEMPORALLY RELATED WINDOWS
# ============================================================

def group_incident_episodes(
    incidents
):

    if not incidents:
        return []


    episodes = []

    current = [
        incidents[0]
    ]


    current_end = incidents[0]["end"]


    for incident in incidents[1:]:

        start = incident["start"]


        # ----------------------------------------------------
        # If this window starts before or shortly after the
        # previous incident episode ends, consider it related.
        # ----------------------------------------------------

        if start <= (
            current_end
            + TEMPORAL_GAP
        ):

            current.append(
                incident
            )

            current_end = max(
                current_end,
                incident["end"]
            )

        else:

            episodes.append(
                current
            )

            current = [
                incident
            ]

            current_end = (
                incident["end"]
            )


    episodes.append(
        current
    )

    return episodes


# ============================================================
# CLASSIFY AN INCIDENT EPISODE
# ============================================================

def classify_episode(
    episode
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We use the earliest confirmed incident as the PRIMARY
    # event.
    #
    # Later classifications are retained as secondary
    # observations instead of being allowed to overwrite it.
    # --------------------------------------------------------

    primary = episode[0]


    primary_class = (
        primary["classification"]
    )


    # --------------------------------------------------------
    # Collect all observations
    # --------------------------------------------------------

    observations = []

    for item in episode:

        classification = (
            item["classification"]
        )

        if classification not in observations:

            observations.append(
                classification
            )


    # --------------------------------------------------------
    # Temporal span
    # --------------------------------------------------------

    start = min(
        item["start"]
        for item in episode
    )

    end = max(
        item["end"]
        for item in episode
    )


    return {

        "start_seconds":
            start,

        "end_seconds":
            end,

        "primary_classification":
            primary_class,

        "observations":
            observations,

        "window_indices":
            [
                item["window_index"]
                for item in episode
            ],

        "window_classifications":
            [
                item["classification"]
                for item in episode
            ],

        "primary_evidence":
            primary[
                primary_class.lower()
                .replace(
                    " ",
                    "_"
                )
            ]["evidence"]
            if primary_class != "Road Accident"
            else primary[
                "road_accident"
            ]["evidence"],
    }


# ============================================================
# AGGREGATE
# ============================================================

def aggregate(data):

    windows = data[
        "windows"
    ]


    incidents = (
        get_incident_windows(
            windows
        )
    )


    episodes = (
        group_incident_episodes(
            incidents
        )
    )


    episode_results = []


    for episode in episodes:

        episode_results.append(
            classify_episode(
                episode
            )
        )


    # --------------------------------------------------------
    # FINAL CLASSIFICATION
    # --------------------------------------------------------

    if not episode_results:

        final_classification = (
            NORMAL_CLASS
        )

    else:

        # Earliest incident episode is treated as the primary
        # video incident.
        final_classification = (
            episode_results[0][
                "primary_classification"
            ]
        )


    # --------------------------------------------------------
    # Count raw window detections
    # --------------------------------------------------------

    raw_counts = {
        "Normal": 0,
        "Fire": 0,
        "Fight": 0,
        "Road Accident": 0,
        "Multiple": 0,
    }


    for window in windows:

        classification = (
            window["classification"]
        )

        if classification in raw_counts:

            raw_counts[
                classification
            ] += 1


    # --------------------------------------------------------
    # Episode-level counts
    # --------------------------------------------------------

    episode_counts = {
        "Fire": 0,
        "Fight": 0,
        "Road Accident": 0,
    }


    for episode in episode_results:

        classification = (
            episode[
                "primary_classification"
            ]
        )

        if classification in episode_counts:

            episode_counts[
                classification
            ] += 1


    return {

        "video":
            data.get("video"),

        "model":
            data.get("model"),

        "duration_seconds":
            data.get(
                "duration_seconds"
            ),

        "window_seconds":
            data.get(
                "window_seconds"
            ),

        "overlap_seconds":
            data.get(
                "overlap_seconds"
            ),

        "raw_window_counts":
            raw_counts,

        "episode_counts":
            episode_counts,

        "number_of_incident_episodes":
            len(episode_results),

        "final_classification":
            final_classification,

        "incident_episodes":
            episode_results,

        "aggregation_method":
            (
                "Temporal episode grouping. "
                "Overlapping/adjacent incident "
                "windows are grouped together. "
                "The earliest confirmed incident "
                "is treated as the primary event; "
                "later labels are retained as "
                "secondary observations."
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python src\\aggregate_results.py "
            "\"path_to_scan_json\""
        )

        sys.exit(1)


    input_path = Path(
        sys.argv[1]
    )


    if not input_path.exists():

        print(
            f"File not found:\n"
            f"{input_path}"
        )

        sys.exit(1)


    print("=" * 70)

    print(
        "SENTINELAI - TEMPORAL EPISODE AGGREGATION"
    )

    print("=" * 70)


    data = load_results(
        input_path
    )


    result = aggregate(
        data
    )


    print()

    print(
        "Video:",
        result["video"]
    )

    print()

    print(
        "Raw window counts:"
    )

    for key, value in (
        result[
            "raw_window_counts"
        ].items()
    ):

        print(
            f"  {key:<15} {value}"
        )


    print()

    print(
        "Incident episodes:",
        result[
            "number_of_incident_episodes"
        ]
    )


    for index, episode in enumerate(
        result[
            "incident_episodes"
        ],
        start=1
    ):

        print()

        print(
            f"Episode {index}:"
        )

        print(
            f"  Time: "
            f"{episode['start_seconds']:.1f}s "
            f"→ "
            f"{episode['end_seconds']:.1f}s"
        )

        print(
            "  Primary:",
            episode[
                "primary_classification"
            ]
        )

        print(
            "  Observations:",
            ", ".join(
                episode[
                    "observations"
                ]
            )
        )

        print(
            "  Windows:",
            episode[
                "window_indices"
            ]
        )

        print(
            "  Evidence:",
            episode[
                "primary_evidence"
            ]
        )


    print()

    print("=" * 70)

    print(
        "FINAL CLASSIFICATION:",
        result[
            "final_classification"
        ]
    )

    print("=" * 70)


    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        input_path.parent
        /
        (
            input_path.stem
            +
            "_aggregated.json"
        )
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )


    print()

    print(
        "Saved:",
        output_path
    )


# ============================================================

if __name__ == "__main__":

    main()