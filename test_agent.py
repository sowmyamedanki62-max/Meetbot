
from services.meeting_agent import generate_meeting_minutes

sample_transcript = """
The team discussed the website launch.
Ravi will finish the frontend by Friday.
Priya will test the application on Thursday.
The team agreed to review the final version on Friday afternoon.
"""

try:
    result = generate_meeting_minutes(sample_transcript)
    print("\n===== MEETBOT MEETING MINUTES =====\n")
    print(result)
except Exception as error:
    print("Test failed:", error)
