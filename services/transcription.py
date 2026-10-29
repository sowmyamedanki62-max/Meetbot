
import os
import tempfile
from pathlib import Path

import whisper

# Locate FFmpeg installed by Winget.
FFMPEG_BIN = (
    Path(os.environ["LOCALAPPDATA"])
    / "Microsoft"
    / "WinGet"
    / "Packages"
    / "Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
    / "ffmpeg-9.0.2-full_build-shared"
    / "bin"
)

# Make FFmpeg accessible to Whisper.
if (FFMPEG_BIN / "ffmpeg.exe").is_file():
    os.environ["PATH"] = (
        str(FFMPEG_BIN)
        + os.pathsep
        + os.environ.get("PATH", "")
    )
else:
    raise FileNotFoundError(
        f"FFmpeg was not found at: {FFMPEG_BIN}"
    )


def transcribe_audio(uploaded_file):
    """Transcribe an uploaded audio file using Whisper."""

    if uploaded_file is None:
        raise ValueError("Please upload an audio file first.")

    temp_path = None

    try:
        suffix = Path(uploaded_file.name).suffix.lower()

        allowed_extensions = {
            ".mp3", ".wav", ".m4a", ".mpeg",
            ".mpga", ".mp4", ".webm"
        }

        if suffix not in allowed_extensions:
            raise ValueError(
                "Unsupported file format. Please upload MP3, WAV, "
                "M4A, MPEG, MPGA, MP4, or WEBM."
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        model = whisper.load_model("tiny")
        result = model.transcribe(temp_path)

        transcript = result.get("text", "").strip()

        if not transcript:
            raise ValueError(
                "No speech was detected. Please try a clearer recording."
            )

        return transcript

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
