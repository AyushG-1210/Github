import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import re

from whisper_back import transcribe_audio
from llm_back import ask_llm
from tts_back import speak


# --- Exit command detection ---
_FILLER = {"please", "now", "hey", "ok", "okay", "computer", "jarvis", "friday", "ai"}
_EXIT_PHRASES = {
    "exit", "quit", "stop", "bye", "goodbye",
    "shut down", "power off", "stop listening"
}

def is_exit_command(text: str) -> bool:
    if not text:
        return False
    s = re.sub(r"[^\w\s]", "", text.lower()).strip()
    tokens = [t for t in s.split() if t not in _FILLER]
    if not tokens:
        return False
    cleaned = " ".join(tokens)
    return cleaned in _EXIT_PHRASES


# --- Audio recording helper ---
def record_audio(filename="input.wav", samplerate=16000):
    print("🎤 Speak now (press ENTER to stop)...")
    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(f"⚠️ Recording issue: {status}")
        recording.append(indata.copy())

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="int16",
        callback=callback
    )
    with stream:
        input()  # wait for ENTER key
    audio = np.concatenate(recording, axis=0)
    write(filename, samplerate, audio)
    print(f"✅ Saved {filename}")
    return filename


# --- Main loop ---
def main():
    while True:
        input_file = record_audio()
        transcript = transcribe_audio(input_file)
        print(f"User: {transcript}")

        if is_exit_command(transcript):
            print("👋 Exiting...")
            break

        reply = ask_llm(transcript)  # already capped in llm_back
        print(f"AI: {reply}")

        output_file = speak(reply)
        print(f"💬 Spoke reply to {output_file}")


if __name__ == "__main__":
    main()
