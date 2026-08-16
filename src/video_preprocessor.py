# ============================================================
# SENTINELAI - SAFE VIDEO PREPROCESSOR
# ============================================================
#
# Purpose:
#   Prepare arbitrary CCTV/video input safely for Qwen2.5-VL.
#
# Features:
#   - Inspects resolution / FPS / frame count / duration
#   - Does NOT upscale small videos
#   - Downscales oversized videos while preserving aspect ratio
#   - Samples video temporally
#   - Creates fixed-duration windows
#   - Hard-limits frames per window
#   - Processes frames sequentially
#   - Does NOT load the whole video into RAM
#   - Creates a manifest.json
#
# Default:
#   Window          = 10 seconds
#   Target FPS      = 2 FPS
#   Max frames      = 20/window
#   Max resolution  = 1280x720
#
# No FFmpeg command-line dependency.
# Uses OpenCV.
# ============================================================

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


# ============================================================
# OpenCV
# ============================================================

import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\SentinelAI_Qwen"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "preprocessed"
)


# ============================================================
# SAFE DEFAULTS
# ============================================================

DEFAULT_WINDOW_SECONDS = 10.0

DEFAULT_TARGET_FPS = 2.0

DEFAULT_MAX_FRAMES_PER_WINDOW = 20

DEFAULT_MAX_WIDTH = 1280

DEFAULT_MAX_HEIGHT = 720


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class VideoInfo:

    input_path: str

    filename: str

    width: int

    height: int

    fps: float

    frame_count: int

    duration: float

    codec: str


@dataclass
class PreprocessedWindow:

    clip_id: str

    clip_path: str

    source_video: str

    start: float

    end: float

    duration: float

    fps: float

    frames: int

    width: int

    height: int


@dataclass
class PreprocessingResult:

    source: VideoInfo

    output_directory: str

    window_seconds: float

    target_fps: float

    frames_per_window: int

    max_frames_per_window: int

    output_width: int

    output_height: int

    windows: List[PreprocessedWindow]


# ============================================================
# VIDEO PREPROCESSOR
# ============================================================

class VideoPreprocessor:

    def __init__(
        self,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        target_fps: float = DEFAULT_TARGET_FPS,
        max_frames_per_window: int = (
            DEFAULT_MAX_FRAMES_PER_WINDOW
        ),
        max_width: int = DEFAULT_MAX_WIDTH,
        max_height: int = DEFAULT_MAX_HEIGHT,
    ):

        # ----------------------------------------------------
        # Validate configuration.
        # ----------------------------------------------------

        if window_seconds <= 0:

            raise ValueError(
                "window_seconds must be greater than 0."
            )

        if target_fps <= 0:

            raise ValueError(
                "target_fps must be greater than 0."
            )

        if max_frames_per_window <= 0:

            raise ValueError(
                "max_frames_per_window must be greater than 0."
            )

        if max_width <= 0:

            raise ValueError(
                "max_width must be greater than 0."
            )

        if max_height <= 0:

            raise ValueError(
                "max_height must be greater than 0."
            )

        # ----------------------------------------------------
        # Store configuration.
        # ----------------------------------------------------

        self.output_root = Path(
            output_root
        )

        self.window_seconds = float(
            window_seconds
        )

        self.max_frames_per_window = int(
            max_frames_per_window
        )

        self.max_width = int(
            max_width
        )

        self.max_height = int(
            max_height
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # These were missing in the previous version.
        # They are not actually needed for upscaling, but
        # we keep them as safety limits.
        # ----------------------------------------------------

        self.min_width = 320

        self.min_height = 240

        # ----------------------------------------------------
        # Make sure FPS can never produce more than the
        # configured number of frames in one window.
        # ----------------------------------------------------

        safe_max_fps = (
            self.max_frames_per_window
            / self.window_seconds
        )

        self.effective_fps = min(
            float(target_fps),
            safe_max_fps,
        )

        # ----------------------------------------------------
        # Number of frames produced for a full window.
        # ----------------------------------------------------

        self.frames_per_window = max(
            1,
            min(
                self.max_frames_per_window,
                int(
                    round(
                        self.window_seconds
                        * self.effective_fps
                    )
                ),
            ),
        )


    # ========================================================
    # INSPECT VIDEO
    # ========================================================

    def inspect(
        self,
        video_path: str | Path,
    ) -> VideoInfo:

        path = Path(
            video_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Video does not exist:\n{path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Input is not a file:\n{path}"
            )

        cap = cv2.VideoCapture(
            str(path)
        )

        if not cap.isOpened():

            cap.release()

            raise RuntimeError(
                "OpenCV could not open the video:\n"
                f"{path}"
            )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fourcc = int(
            cap.get(
                cv2.CAP_PROP_FOURCC
            )
        )

        codec = ""

        if fourcc:

            try:

                codec = "".join(
                    [
                        chr(
                            fourcc
                            & 0xFF
                        ),
                        chr(
                            (fourcc >> 8)
                            & 0xFF
                        ),
                        chr(
                            (fourcc >> 16)
                            & 0xFF
                        ),
                        chr(
                            (fourcc >> 24)
                            & 0xFF
                        ),
                    ]
                )

            except Exception:

                codec = "unknown"

        cap.release()

        # ----------------------------------------------------
        # Some codecs return 0/NaN FPS.
        # ----------------------------------------------------

        if (
            not math.isfinite(fps)
            or fps <= 0
        ):

            fps = 30.0

        if width <= 0:

            raise RuntimeError(
                "Invalid video width."
            )

        if height <= 0:

            raise RuntimeError(
                "Invalid video height."
            )

        if frame_count <= 0:

            raise RuntimeError(
                "OpenCV could not determine frame count."
            )

        duration = (
            frame_count
            / fps
        )

        if duration <= 0:

            raise RuntimeError(
                "Invalid video duration."
            )

        return VideoInfo(
            input_path=str(path),
            filename=path.name,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec,
        )


    # ========================================================
    # CALCULATE SAFE OUTPUT RESOLUTION
    # ========================================================

    def calculate_output_size(
        self,
        width: int,
        height: int,
    ):

        # ----------------------------------------------------
        # NEVER UPSCALE.
        #
        # 320x240 stays 320x240.
        # 640x480 stays 640x480.
        #
        # Large video gets downscaled.
        # ----------------------------------------------------

        scale = min(
            1.0,
            self.max_width / width,
            self.max_height / height,
        )

        new_width = int(
            round(
                width * scale
            )
        )

        new_height = int(
            round(
                height * scale
            )
        )

        new_width = max(
            1,
            new_width,
        )

        new_height = max(
            1,
            new_height,
        )

        # ----------------------------------------------------
        # Ensure even dimensions for video codecs.
        # ----------------------------------------------------

        if new_width > 1:

            new_width -= (
                new_width % 2
            )

        if new_height > 1:

            new_height -= (
                new_height % 2
            )

        new_width = max(
            2,
            new_width,
        )

        new_height = max(
            2,
            new_height,
        )

        return (
            new_width,
            new_height,
        )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    def prepare_output_directory(
        self,
        source_path: Path,
        clean: bool = True,
    ):

        output_dir = (
            self.output_root
            / source_path.stem
        )

        if (
            clean
            and output_dir.exists()
        ):

            shutil.rmtree(
                output_dir
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output_dir


    # ========================================================
    # CREATE VIDEO WRITER
    # ========================================================

    def create_writer(
        self,
        output_path: Path,
        width: int,
        height: int,
    ):

        # ----------------------------------------------------
        # mp4v is used so the script does not require an
        # ffmpeg executable in PATH.
        # ----------------------------------------------------

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.effective_fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():

            writer.release()

            raise RuntimeError(
                "OpenCV could not create output MP4:\n"
                f"{output_path}\n\n"
                "Your OpenCV build may not have a working "
                "video encoder."
            )

        return writer


    # ========================================================
    # WRITE ONE WINDOW
    # ========================================================

    def _write_window(
        self,
        frames,
        window_index: int,
        start_time: float,
        end_time: float,
        source_path: Path,
        output_dir: Path,
        output_width: int,
        output_height: int,
    ):

        if not frames:

            return None

        # ----------------------------------------------------
        # Hard safety limit.
        # ----------------------------------------------------

        if len(
            frames
        ) > self.max_frames_per_window:

            frames = frames[
                :self.max_frames_per_window
            ]

        clip_id = (
            f"{source_path.stem}"
            f"_{window_index:04d}"
        )

        clip_path = (
            output_dir
            / f"{clip_id}.mp4"
        )

        writer = self.create_writer(
            clip_path,
            output_width,
            output_height,
        )

        written = 0

        try:

            for frame in frames:

                # ------------------------------------------------
                # Downscale only when necessary.
                # ------------------------------------------------

                if (
                    frame.shape[1]
                    != output_width
                    or
                    frame.shape[0]
                    != output_height
                ):

                    frame = cv2.resize(
                        frame,
                        (
                            output_width,
                            output_height,
                        ),
                        interpolation=(
                            cv2.INTER_AREA
                        ),
                    )

                writer.write(
                    frame
                )

                written += 1

        finally:

            writer.release()

        if written == 0:

            if clip_path.exists():

                clip_path.unlink()

            return None

        return (
            PreprocessedWindow(
                clip_id=clip_id,
                clip_path=str(
                    clip_path
                ),
                source_video=str(
                    source_path
                ),
                start=float(
                    start_time
                ),
                end=float(
                    end_time
                ),
                duration=float(
                    end_time
                    - start_time
                ),
                fps=float(
                    self.effective_fps
                ),
                frames=written,
                width=output_width,
                height=output_height,
            )
        )


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    def process(
        self,
        video_path: str | Path,
        clean: bool = True,
    ) -> PreprocessingResult:

        source_path = Path(
            video_path
        )

        print()
        print("=" * 70)
        print(
            "SENTINELAI - VIDEO PREPROCESSING"
        )
        print("=" * 70)

        print()
        print(
            "Input:"
        )

        print(
            source_path
        )

        # ----------------------------------------------------
        # Inspect.
        # ----------------------------------------------------

        info = self.inspect(
            source_path
        )

        print()
        print(
            "INPUT VIDEO"
        )

        print(
            f"Resolution: "
            f"{info.width}x{info.height}"
        )

        print(
            f"FPS: "
            f"{info.fps:.3f}"
        )

        print(
            f"Frames: "
            f"{info.frame_count}"
        )

        print(
            f"Duration: "
            f"{info.duration:.2f}s"
        )

        print(
            f"Codec: "
            f"{info.codec or 'unknown'}"
        )

        # ----------------------------------------------------
        # Safe output dimensions.
        # ----------------------------------------------------

        output_width, output_height = (
            self.calculate_output_size(
                info.width,
                info.height,
            )
        )

        print()
        print(
            "NORMALIZATION"
        )

        print(
            f"Output resolution: "
            f"{output_width}x{output_height}"
        )

        print(
            f"Output FPS: "
            f"{self.effective_fps:.3f}"
        )

        print(
            f"Window duration: "
            f"{self.window_seconds:.2f}s"
        )

        print(
            f"Frames/window: "
            f"{self.frames_per_window}"
        )

        print(
            f"Maximum frames/window: "
            f"{self.max_frames_per_window}"
        )

        # ----------------------------------------------------
        # Prepare output.
        # ----------------------------------------------------

        output_dir = (
            self.prepare_output_directory(
                source_path,
                clean=clean,
            )
        )

        # ----------------------------------------------------
        # Open source.
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            str(source_path)
        )

        if not cap.isOpened():

            cap.release()

            raise RuntimeError(
                f"Could not reopen video:\n"
                f"{source_path}"
            )

        source_fps = info.fps

        total_duration = info.duration

        # ----------------------------------------------------
        # Number of temporal windows.
        #
        # Example:
        # 21.57 sec
        #
        # -> 0-10
        # -> 10-20
        # -> 20-21.57
        # ----------------------------------------------------

        window_count = int(
            math.ceil(
                total_duration
                / self.window_seconds
            )
        )

        windows = []

        # ----------------------------------------------------
        # We process the video sequentially.
        #
        # Only the current window's sampled frames are
        # stored in memory.
        # ----------------------------------------------------

        current_window_index = 0

        current_window_start = 0.0

        current_window_end = min(
            self.window_seconds,
            total_duration,
        )

        current_window_frames = []

        next_sample_time = 0.0

        source_frame_index = 0

        # ----------------------------------------------------
        # Read source frames.
        # ----------------------------------------------------

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_time = (
                source_frame_index
                / source_fps
            )

            source_frame_index += 1

            # ------------------------------------------------
            # Skip windows if necessary.
            # ------------------------------------------------

            while (
                frame_time
                >= current_window_end
                and current_window_index
                < window_count
            ):

                result = (
                    self._write_window(
                        current_window_frames,
                        current_window_index,
                        current_window_start,
                        current_window_end,
                        source_path,
                        output_dir,
                        output_width,
                        output_height,
                    )
                )

                if result is not None:

                    windows.append(
                        result
                    )

                    print(
                        f"  Created "
                        f"{result.clip_id}: "
                        f"{result.start:.2f}s -> "
                        f"{result.end:.2f}s | "
                        f"{result.frames} frames"
                    )

                current_window_index += 1

                current_window_frames = []

                current_window_start = (
                    current_window_index
                    * self.window_seconds
                )

                current_window_end = min(
                    current_window_start
                    + self.window_seconds,
                    total_duration,
                )

                # ------------------------------------------------
                # Critical:
                #
                # Reset sampling schedule for every new window.
                # ------------------------------------------------

                next_sample_time = (
                    current_window_start
                )

            if (
                current_window_index
                >= window_count
            ):

                break

            # ------------------------------------------------
            # Only sample frames inside the current window.
            # ------------------------------------------------

            if (
                frame_time
                < current_window_start
            ):

                continue

            if (
                frame_time
                >= current_window_end
            ):

                continue

            # ------------------------------------------------
            # Temporal sampling.
            #
            # Example at 2 FPS:
            #
            # 0.0
            # 0.5
            # 1.0
            # 1.5
            # ...
            #
            # Maximum 20 frames/window.
            # ------------------------------------------------

            if (
                frame_time
                + (
                    0.5
                    / source_fps
                )
                >= next_sample_time
            ):

                if len(
                    current_window_frames
                ) < self.max_frames_per_window:

                    current_window_frames.append(
                        frame.copy()
                    )

                next_sample_time += (
                    1.0
                    / self.effective_fps
                )

        # ----------------------------------------------------
        # Flush final window.
        # ----------------------------------------------------

        if (
            current_window_frames
            and current_window_index
            < window_count
        ):

            result = (
                self._write_window(
                    current_window_frames,
                    current_window_index,
                    current_window_start,
                    current_window_end,
                    source_path,
                    output_dir,
                    output_width,
                    output_height,
                )
            )

            if result is not None:

                windows.append(
                    result
                )

                print(
                    f"  Created "
                    f"{result.clip_id}: "
                    f"{result.start:.2f}s -> "
                    f"{result.end:.2f}s | "
                    f"{result.frames} frames"
                )

        cap.release()

        # ----------------------------------------------------
        # Manifest.
        # ----------------------------------------------------

        result = (
            PreprocessingResult(
                source=info,
                output_directory=str(
                    output_dir
                ),
                window_seconds=(
                    self.window_seconds
                ),
                target_fps=(
                    self.effective_fps
                ),
                frames_per_window=(
                    self.frames_per_window
                ),
                max_frames_per_window=(
                    self.max_frames_per_window
                ),
                output_width=(
                    output_width
                ),
                output_height=(
                    output_height
                ),
                windows=windows,
            )
        )

        manifest_path = (
            output_dir
            / "manifest.json"
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                asdict(result),
                f,
                indent=2,
                ensure_ascii=False,
            )

        # ----------------------------------------------------
        # Final report.
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "PREPROCESSING COMPLETE"
        )
        print("=" * 70)

        print()
        print(
            f"Original resolution: "
            f"{info.width}x{info.height}"
        )

        print(
            f"Output resolution: "
            f"{output_width}x{output_height}"
        )

        print(
            f"Original FPS: "
            f"{info.fps:.2f}"
        )

        print(
            f"Output FPS: "
            f"{self.effective_fps:.2f}"
        )

        print(
            f"Duration: "
            f"{info.duration:.2f}s"
        )

        print(
            f"Windows created: "
            f"{len(windows)}"
        )

        print()
        print(
            "Output directory:"
        )

        print(
            output_dir
        )

        print()
        print(
            "Manifest:"
        )

        print(
            manifest_path
        )

        print("=" * 70)

        return result


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def preprocess_video(
    video_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
):

    processor = VideoPreprocessor(
        output_root=Path(
            output_root
        )
    )

    return processor.process(
        video_path
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "SentinelAI safe video preprocessing "
            "for Qwen2.5-VL."
        )
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to input video.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
        help=(
            "Output directory. "
            "Default: "
            "C:\\SentinelAI_Qwen\\dataset\\preprocessed"
        ),
    )

    parser.add_argument(
        "--window",
        type=float,
        default=10.0,
        help=(
            "Window duration in seconds. "
            "Default: 10."
        ),
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help=(
            "Output sampling FPS. "
            "Default: 2."
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help=(
            "Maximum frames per window. "
            "Default: 20."
        ),
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help=(
            "Maximum output width. "
            "Default: 1280."
        ),
    )

    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help=(
            "Maximum output height. "
            "Default: 720."
        ),
    )

    args = parser.parse_args()

    processor = VideoPreprocessor(
        output_root=Path(
            args.output
        ),
        window_seconds=(
            args.window
        ),
        target_fps=(
            args.fps
        ),
        max_frames_per_window=(
            args.max_frames
        ),
        max_width=(
            args.max_width
        ),
        max_height=(
            args.max_height
        ),
    )

    processor.process(
        args.video
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()