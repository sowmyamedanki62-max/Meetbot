import os
import shutil
import tempfile
from pathlib import Path

import whisper


def configure_ffmpeg():
    """Find FFmpeg on Windows or Linux."""

    # First, check whether FFmpeg is already available.
    if shutil.which("ffmpeg"):
        return

    # Windows: check the Winget installation path.
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")

        if local_app_data:
            ffmpeg_bin = (
                Path(local_app_data)
                / "Microsoft"
                / "WinGet"
                / "Packages"
                / "Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
                / "ffmpeg-9.0.2-full_build-shared"
                / "bin"
            )

            if (ffmpeg_bin / "ffmpeg.exe").is_file():
                os.environ["PATH"] = (
                    str(ffmpeg_bin)
                    + os.pathsep
                    + os.environ.get("PATH", "")
                )
                return

    raise FileNotFoundError(
        "FFmpeg was not found. Install FFmpeg and make sure "
        "the ffmpeg executable is available on PATH."
    )


def transcribe_audio(uploaded_file):
    """Transcribe an uploaded audio file using Whisper."""

    if uploaded_file is None:
        raise ValueError("Please upload an audio file first.")

    temp_path = None

    try:
        suffix = Path(uploaded_file.name).suffix.lower()

        allowed_extensions = {
            ".mp3",
            ".wav",
            ".m4a",
            ".mpeg",
            ".mpga",
            ".mp4",
            ".webm",
        }

        if suffix not in allowed_extensions:
            raise ValueError(
                "Unsupported file format. Please upload MP3, WAV, "
                "M4A, MPEG, MPGA, MP4, or WEBM."
            )

        configure_ffmpeg()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
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
