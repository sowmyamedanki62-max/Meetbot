from gtts import gTTS

meeting_text = """
Good morning, everyone. Today we discussed the progress of our website project.
Ravi will complete the frontend development by Friday.
Priya will test the application on Thursday and report any bugs.
The team agreed to review the final website on Friday afternoon.
We also discussed improving the login page and fixing the mobile layout.
Ravi will check the mobile layout after completing the frontend.
Our next review meeting will be on Monday. Thank you, everyone.
"""

audio = gTTS(text=meeting_text, lang="en")
audio.save("sample_meeting.mp3")

print("Sample meeting audio created successfully!")