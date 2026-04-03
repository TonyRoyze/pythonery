#!/usr/bin/env python3
"""
audio_trim.py — Trim / cut / merge / speed-adjust audio from a YAML schema.

Usage:
    python audio_trim.py                          # uses audio_cut_schema.yaml
    python audio_trim.py my_schema.yaml
    python audio_trim.py --schema my_schema.yaml

Dependencies:
    pip install pydub pyyaml
    brew install ffmpeg        # must be on PATH
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency: pip install pyyaml")

try:
    from pydub import AudioSegment
    from pydub.effects import normalize
except ImportError:
    sys.exit("Missing dependency: pip install pydub  (and brew install ffmpeg)")


# ──────────────────────────────────────────────
#  Timestamp parser
# ──────────────────────────────────────────────

def _ts_to_ms(value) -> int:
    """
    Convert a timestamp to milliseconds.
    Accepted formats:
      - int / float      → treated as seconds  (90 → 90 000 ms)
      - "SS"             → seconds
      - "MM:SS"          → minutes + seconds
      - "HH:MM:SS"       → hours + minutes + seconds
      - "HH:MM:SS.mmm"   → millisecond precision
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value * 1000)

    s = str(value).strip()
    ms_extra = 0
    if "." in s:
        s, ms_str = s.rsplit(".", 1)
        ms_extra = int(ms_str.ljust(3, "0")[:3])

    parts = s.split(":")
    if len(parts) == 1:
        total_s = int(parts[0])
    elif len(parts) == 2:
        total_s = int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        total_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        raise ValueError(f"Cannot parse timestamp: '{value}'")

    return total_s * 1000 + ms_extra


def _ms_label(ms: int) -> str:
    """Format milliseconds as MM:SS.mmm for display."""
    total_s, rem_ms = divmod(ms, 1000)
    m, s = divmod(total_s, 60)
    return f"{m:02}:{s:02}.{rem_ms:03}"


# ──────────────────────────────────────────────
#  Speed change via ffmpeg atempo
# ──────────────────────────────────────────────

def _atempo_chain(speed: float) -> str:
    """
    Build a chained atempo filter string.
    Each stage must be in [0.5, 2.0]; chain as many as needed.
    e.g. speed=4.0  →  atempo=2.0,atempo=2.0
         speed=0.25 →  atempo=0.5,atempo=0.5
    """
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def change_speed(chunk: AudioSegment, speed: float) -> AudioSegment:
    """
    Pitch-preserving time-stretch via ffmpeg's atempo filter.
    speed > 1.0 → faster,  speed < 1.0 → slower.
    """
    if abs(speed - 1.0) < 1e-6:
        return chunk

    filter_str = _atempo_chain(speed)

    tmp_in  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_in.close()
    tmp_out.close()

    try:
        chunk.export(tmp_in.name, format="wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_in.name,
                "-filter:a", filter_str,
                tmp_out.name,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result = AudioSegment.from_wav(tmp_out.name)
    finally:
        os.unlink(tmp_in.name)
        os.unlink(tmp_out.name)

    return result


# ──────────────────────────────────────────────
#  Schema loader & validator
# ──────────────────────────────────────────────

def load_schema(schema_path: Path) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    for key in ("input", "output", "segments"):
        if key not in schema:
            sys.exit(f"Schema error: missing top-level key '{key}'")

    if "file" not in schema["input"]:
        sys.exit("Schema error: 'input.file' is required")

    if not schema.get("segments"):
        sys.exit("Schema error: 'segments' list is empty")

    for i, seg in enumerate(schema["segments"]):
        if "start" not in seg:
            sys.exit(f"Schema error: segment [{i}] is missing 'start'")
        if "name" not in seg:
            seg["name"] = f"segment_{i + 1}"

    return schema


# ──────────────────────────────────────────────
#  Core processing
# ──────────────────────────────────────────────

def process(schema_path: Path):
    schema     = load_schema(schema_path)
    schema_dir = schema_path.parent

    # ── resolve input file ──
    input_file = Path(schema["input"]["file"])
    if not input_file.is_absolute():
        input_file = schema_dir / input_file
    if not input_file.exists():
        sys.exit(f"Input file not found: {input_file}")

    # ── resolve output dir ──
    out_cfg     = schema.get("output", {})
    out_dir     = Path(out_cfg.get("dir", "./output"))
    if not out_dir.is_absolute():
        out_dir = schema_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fmt     = out_cfg.get("format", "mp3").lower()
    out_bitrate = out_cfg.get("bitrate", "192k")

    # ── global defaults ──
    defaults     = schema.get("defaults", {})
    def_fade_in  = int(defaults.get("fade_in_ms",  0))
    def_fade_out = int(defaults.get("fade_out_ms", 0))
    def_norm     = bool(defaults.get("normalize",   False))
    def_loop     = max(1, int(defaults.get("loop",  1)))
    def_speed    = float(defaults.get("speed",      1.0))

    # ── load audio ──
    print(f"\n▶  Loading  : {input_file.name}")
    audio    = AudioSegment.from_file(str(input_file))
    total_ms = len(audio)
    print(f"   Duration : {total_ms / 1000:.2f}s  ({total_ms} ms)\n")

    # ── process each segment ──
    segments      = schema["segments"]
    pad           = len(str(len(segments)))
    built_chunks  = {}   # name → processed AudioSegment, for use in merge

    print("── Segments " + "─" * 50)
    for idx, seg in enumerate(segments, start=1):
        name     = seg["name"]
        start_ms = _ts_to_ms(seg["start"])
        end_ms   = _ts_to_ms(seg.get("end"))
        fade_in  = int(seg.get("fade_in_ms",  def_fade_in))
        fade_out = int(seg.get("fade_out_ms", def_fade_out))
        do_norm  = bool(seg.get("normalize",   def_norm))
        loop     = max(1, int(seg.get("loop",  def_loop)))
        speed    = float(seg.get("speed",      def_speed))

        if end_ms is None:
            end_ms = total_ms

        # Bounds checking
        if start_ms < 0 or start_ms >= total_ms:
            print(f"  ⚠  [{idx:0{pad}}] '{name}'  — start out of range, skipping.")
            continue
        if end_ms > total_ms:
            print(f"  ⚠  [{idx:0{pad}}] '{name}'  — end clamped to EOF.")
            end_ms = total_ms
        if start_ms >= end_ms:
            print(f"  ⚠  [{idx:0{pad}}] '{name}'  — start ≥ end, skipping.")
            continue

        # Slice → loop → speed → fades → normalize
        chunk = audio[start_ms:end_ms]

        if loop > 1:
            chunk = chunk * loop

        if abs(speed - 1.0) > 1e-6:
            print(f"     applying speed ×{speed} …", end="\r")
            chunk = change_speed(chunk, speed)

        if fade_in > 0:
            chunk = chunk.fade_in(min(fade_in, len(chunk)))
        if fade_out > 0:
            chunk = chunk.fade_out(min(fade_out, len(chunk)))

        if do_norm:
            chunk = normalize(chunk)

        # Save chunk for potential merge later
        built_chunks[name] = chunk

        # Export individual file
        safe_name    = re.sub(r'[\\/*?:"<>|]', "_", name)
        out_file     = out_dir / f"{idx:0{pad}}_{safe_name}.{out_fmt}"
        export_kwargs = {"format": out_fmt}
        if out_fmt in ("mp3", "aac", "ogg"):
            export_kwargs["bitrate"] = out_bitrate
        chunk.export(str(out_file), **export_kwargs)

        segment_s  = (end_ms - start_ms) / 1000
        duration_s = len(chunk) / 1000          # actual length after speed change
        loop_label  = f" ×{loop}"    if loop  > 1         else ""
        speed_label = f" @{speed}×"  if abs(speed - 1.0) > 1e-6 else ""
        print(
            f"  ✔  [{idx:0{pad}}] '{name}'{loop_label}{speed_label}  "
            f"{_ms_label(start_ms)} → {_ms_label(end_ms)}  "
            f"({duration_s:.2f}s)  →  {out_file.name}"
        )

    # ── merge blocks ──
    merge_list = schema.get("merge", [])
    if merge_list:
        print("\n── Merges " + "─" * 52)
        for merge_cfg in merge_list:
            merge_name        = merge_cfg.get("name", "merged")
            seg_names         = merge_cfg.get("segments", [])
            crossfade_ms      = int(merge_cfg.get("crossfade_ms", 0))
            merge_fade_in     = int(merge_cfg.get("fade_in_ms",   0))
            merge_fade_out    = int(merge_cfg.get("fade_out_ms",  0))
            merge_norm        = bool(merge_cfg.get("normalize",    False))
            merge_speed       = float(merge_cfg.get("speed",       1.0))

            # Resolve "all" shorthand
            if seg_names == "all" or seg_names == ["all"]:
                seg_names = [s["name"] for s in segments]

            missing = [n for n in seg_names if n not in built_chunks]
            if missing:
                print(f"  ⚠  merge '{merge_name}': unknown segment(s) {missing}, skipping.")
                continue

            # Join
            combined = AudioSegment.empty()
            for i, sname in enumerate(seg_names):
                part = built_chunks[sname]
                if i == 0 or crossfade_ms == 0:
                    combined = combined + part
                else:
                    combined = combined.append(part, crossfade=crossfade_ms)

            # Speed on combined (before fades so endpoints stay clean)
            if abs(merge_speed - 1.0) > 1e-6:
                print(f"     applying speed ×{merge_speed} to '{merge_name}' …", end="\r")
                combined = change_speed(combined, merge_speed)

            # Fades on combined
            if merge_fade_in > 0:
                combined = combined.fade_in(min(merge_fade_in, len(combined)))
            if merge_fade_out > 0:
                combined = combined.fade_out(min(merge_fade_out, len(combined)))
            if merge_norm:
                combined = normalize(combined)

            safe_merge   = re.sub(r'[\\/*?:"<>|]', "_", merge_name)
            merge_file   = out_dir / f"{safe_merge}.{out_fmt}"
            export_kwargs = {"format": out_fmt}
            if out_fmt in ("mp3", "aac", "ogg"):
                export_kwargs["bitrate"] = out_bitrate
            combined.export(str(merge_file), **export_kwargs)

            xf_label    = f"  crossfade {crossfade_ms}ms" if crossfade_ms else ""
            speed_label = f"  @{merge_speed}×" if abs(merge_speed - 1.0) > 1e-6 else ""
            print(
                f"  ✔  '{merge_name}'  [{' + '.join(seg_names)}]{xf_label}{speed_label}  "
                f"({len(combined)/1000:.2f}s)  →  {merge_file.name}"
            )

    print(f"\n✅  Done. Output folder: {out_dir.resolve()}\n")


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trim / cut / merge / speed-adjust audio from a YAML schema."
    )
    parser.add_argument(
        "schema",
        nargs="?",
        default="audio_cut_schema.yaml",
        help="Path to the YAML schema file (default: audio_cut_schema.yaml)",
    )
    parser.add_argument(
        "--schema",
        dest="schema_flag",
        default=None,
        help="Explicit path to the YAML schema file.",
    )
    args = parser.parse_args()

    schema_path = Path(args.schema_flag or args.schema)
    if not schema_path.is_absolute():
        schema_path = Path.cwd() / schema_path
    if not schema_path.exists():
        sys.exit(f"Schema file not found: {schema_path}")

    process(schema_path)


if __name__ == "__main__":
    main()
