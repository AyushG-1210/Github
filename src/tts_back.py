import sounddevice as sd
import numpy as np
from piper import PiperVoice

# Load Piper voice
voice = PiperVoice.load(
    r"C:\Users\anshu\Desktop\Mini Project\Helmet_Env\Github\models\en_US-amy-medium.onnx"
)

SAMPLE_RATE = voice.config.sample_rate

def speak(text):
    """Generate speech and play it directly."""

    # Collect PCM16 chunks
    audio_chunks = [chunk.audio_int16_bytes for chunk in voice.synthesize(text)]

    # Join into one byte stream
    audio_bytes = b"".join(audio_chunks)

    # Convert bytes -> numpy array
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    # Play back
    sd.play(audio_np, samplerate=SAMPLE_RATE)
    sd.wait()

    return f"Played {len(audio_np)} samples in real-time"