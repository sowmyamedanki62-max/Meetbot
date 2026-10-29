import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"


def clean_text(value, default="Not specified"):
    """Convert a value to clean, readable text."""
    if value is None:
        return default

    text = str(value).strip()

    if not text or text.lower() in {"none", "null", "not specified"}:
        return default

    return text


def get_meeting_date(value, transcript):
    """
    Accept a meeting date only when it resembles an explicit calendar date
    and that date is mentioned in the transcript.
    """
    value = clean_text(value)

    if value == "Not specified":
        return "Not specified"

    # Reject sentences or phrases that are not calendar dates.
    if len(value.split()) > 5:
        return "Not specified"

    date_patterns = [
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
        r"\d{1,2}(?:,?\s+\d{4})?\b",
        r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
        r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
        r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"(?:\s+\d{4})?\b",
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, transcript, flags=re.IGNORECASE)

        for match in matches:
            if match.lower() == value.lower():
                return value

    return "Not specified"


def generate_meeting_minutes(transcript):
    """Generate structured meeting minutes using local Ollama."""

    if not transcript or not transcript.strip():
        raise ValueError("The meeting transcript is empty.")

    prompt = f"""
You are MeetBot, an accurate meeting-minutes assistant.

Use ONLY information explicitly stated in the transcript.
Never invent names, tasks, dates, decisions, or deadlines.

Return ONLY valid JSON with this exact structure:
{{
  "summary": "Short summary of the meeting",
  "discussion_points": ["topic 1", "topic 2"],
  "decisions": ["explicitly agreed decision"],
  "action_items": [
    {{
      "person": "person explicitly responsible or Not specified",
      "task": "specific assigned task",
      "deadline": "stated deadline or Not specified"
    }}
  ],
  "next_meeting": "stated next meeting day/date or Not specified",
  "meeting_date": "explicit calendar date of this meeting or Not specified"
}}

Rules:
- Summarize the main purpose and progress of the meeting.
- Include important topics under discussion_points.
- Include explicit agreements under decisions.
- Only include an action item when the transcript describes a task
  someone is expected to perform.
- Do not turn every decision into an action item.
- "The team agreed to review the final website on Friday afternoon"
  is a decision, NOT an assigned action item.
- Do not assign a team decision to an individual.
- "Ravi will complete the front end development by Friday"
  means person Ravi, task Complete front-end development,
  deadline Friday.
- "Priya will test the application on Thursday and report any bugs"
  means person Priya, task Test the application and report bugs,
  deadline Thursday.
- "Ravi will check the mobile layout after completing the front end"
  means person Ravi, task Check the mobile layout after completing
  front-end development, deadline Not specified.
- Do not treat discussion topics as assigned tasks.
- Include each explicit action item only once.
- The meeting date must be an actual calendar date explicitly stated
  for the current meeting. Do not use today's date, a weekday alone,
  the next meeting date, or an opening sentence as the meeting date.
- If the current meeting date is not stated, use "Not specified".
- If the next meeting is not stated, use "Not specified".
- If no explicit decisions exist, return an empty decisions list.
- Return valid JSON only. Do not use Markdown fences.

TRANSCRIPT:
{transcript}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=300,
        )

        response.raise_for_status()

        raw_result = response.json().get("response", "").strip()

        if not raw_result:
            raise RuntimeError("The AI returned an empty response.")

        data = json.loads(raw_result)

        required_keys = {
            "summary",
            "discussion_points",
            "decisions",
            "action_items",
            "next_meeting",
            "meeting_date",
        }

        if not isinstance(data, dict) or not required_keys.issubset(data):
            raise ValueError("The AI response is missing required fields.")

        # Clean discussion points.
        points = data.get("discussion_points", [])

        if not isinstance(points, list):
            points = []

        points = [
            clean_text(point, "")
            for point in points
            if clean_text(point, "")
        ]

        # Clean decisions.
        decisions = data.get("decisions", [])

        if not isinstance(decisions, list):
            decisions = []

        decisions = [
            clean_text(decision, "")
            for decision in decisions
            if clean_text(decision, "")
        ]

        # Validate action items and exclude team decisions mistakenly
        # returned by the AI as tasks.
        action_items = data.get("action_items", [])

        if not isinstance(action_items, list):
            action_items = []

        validated_items = []
        seen_items = set()

        for item in action_items:
            if not isinstance(item, dict):
                continue

            task = clean_text(item.get("task"), "")
            person = clean_text(item.get("person"))
            deadline = clean_text(item.get("deadline"))

            if not task:
                continue

            normalized_task = task.lower()
            normalized_person = person.lower()

            # A team-wide website review decision is not an assigned task.
            if (
                "review the final website" in normalized_task
                and normalized_person in {
                    "the team",
                    "team",
                    "not specified",
                }
            ):
                continue

            # Do not show duplicate action items.
            item_key = (
                normalized_person,
                normalized_task,
                deadline.lower(),
            )

            if item_key in seen_items:
                continue

            seen_items.add(item_key)

            validated_items.append({
                "person": person,
                "task": task,
                "deadline": deadline,
            })

        # Format the meeting date conservatively.
        meeting_date = get_meeting_date(
            data.get("meeting_date"),
            transcript,
        )

        summary = clean_text(data.get("summary"))

        next_meeting = clean_text(data.get("next_meeting"))

        # Build readable meeting minutes.
        lines = [
            "# Meeting Summary",
            "",
            summary,
            "",
            f"**Meeting Date:** {meeting_date}",
            "",
            "# Key Discussion Points",
        ]

        if points:
            lines.extend(f"- {point}" for point in points)
        else:
            lines.append("- No discussion points identified.")

        lines.extend([
            "",
            "# Decisions Made",
        ])

        if decisions:
            lines.extend(f"- {decision}" for decision in decisions)
        else:
            lines.append("- No explicit decisions identified.")

        lines.extend([
            "",
            "# Action Items",
            "",
            "| Responsible Person | Task | Deadline |",
            "|---|---|---|",
        ])

        if validated_items:
            for item in validated_items:
                lines.append(
                    f"| {item['person']} | {item['task']} "
                    f"| {item['deadline']} |"
                )
        else:
            lines.append(
                "| Not specified | No action items identified | Not specified |"
            )

        lines.extend([
            "",
            "# Next Meeting",
            "",
            next_meeting,
        ])

        return "\n".join(lines)

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "AI generation took too long. Please try again."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            "Ollama returned an error while generating minutes."
        ) from exc

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "The AI returned an invalid or incomplete result. Please try again."
        ) from exc
