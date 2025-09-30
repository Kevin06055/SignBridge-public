import whisper

# Load model (base is faster, large is more accurate)
model = whisper.load_model("base")

# Transcribe an audio file
result = model.transcribe("test.mp3")

# Print text
print("Transcription:", result["text"])
