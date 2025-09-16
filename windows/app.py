import asyncio, json, time, queue, configparser
import numpy as np
import sounddevice as sd
import webrtcvad
import requests
import websockets
from faster_whisper import WhisperModel
import edge_tts

# Optional wake word
try:
    from openwakeword.model import Model as WakeModel
    HAS_WAKE = True
except Exception:
    HAS_WAKE = False

cfg = configparser.ConfigParser()
cfg.read("config.ini")

SR = cfg.getint("audio", "sample_rate", fallback=16000)
VAD_LEVEL = cfg.getint("audio", "vad_aggressiveness", fallback=2)
MAX_UTT_MS = cfg.getint("audio", "max_utterance_ms", fallback=8000)
SIL_MS = cfg.getint("audio", "silence_ms", fallback=900)

REQUIRE_WAKE = cfg.getboolean("wake", "require_wake_word", fallback=True)
WAKE_PHRASE = cfg.get("wake", "phrase", fallback="brighton")

SERVER_URL = cfg.get("server", "url", fallback="http://127.0.0.1:8080/chat")
WS_URL = cfg.get("server", "ws", fallback="ws://127.0.0.1:8081")
DEVICE = cfg.get("server", "device", fallback="windows")

VOICE = cfg.get("tts", "voice", fallback="en-US-JennyNeural")

# ---- Arbitration token over WS (non-blocking) ----
class Arbiter:
    def __init__(self):
        self.has_token = False
    async def run(self):
        try:
            async with websockets.connect(WS_URL) as ws:
                await ws.send(json.dumps({"type":"claim","device":DEVICE}))
                msg = json.loads(await ws.recv())
                if msg.get("type") == "granted":
                    self.has_token = True
                # Keep socket alive
                while True:
                    await asyncio.sleep(5)
        except Exception:
            # If WS is down, assume solo use
            self.has_token = True
    def release(self):
        self.has_token = True

arbiter = Arbiter()

# ---- Audio + VAD ----
vad = webrtcvad.Vad(VAD_LEVEL)
frame_ms = 30
frame_len = int(SR * frame_ms / 1000)
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status: pass
    mono = indata.copy()
    if mono.ndim > 1:
        mono = mono[:,0]
    audio_q.put(mono.astype(np.int16))

def is_speech_frame(pcm16):
    try:
        return vad.is_speech(pcm16.tobytes(), SR)
    except Exception:
        return False

# ---- Wake word ----
if HAS_WAKE:
    wake_model = WakeModel()
else:
    wake_model = None

def heard_wake(pcm16):
    if not REQUIRE_WAKE:
        return True
    if not wake_model:
        # MVP fallback: accept as wake
        return True
    f = pcm16.astype(np.float32) / 32768.0
    preds = wake_model.predict([f])
    for _, score in preds.items():
        if score > 0.5:
            return True
    return False

# ---- STT ----
stt_model = WhisperModel("small", device="cpu", compute_type="int8")

async def tts_say(text):
    try:
        communicate = edge_tts.Communicate(text, voice=VOICE)
        async for _ in communicate.stream():
            pass
    except Exception:
        print("TTS:", text)

def record_until_silence(max_ms=MAX_UTT_MS):
    buf = []
    start = time.time()
    silence_start = None
    while True:
        try:
            frame = audio_q.get(timeout=1)
        except queue.Empty:
            break
        buf.append(frame)
        pcm = buf[-1]
        if is_speech_frame(pcm):
            silence_start = None
        else:
            if silence_start is None:
                silence_start = time.time()
            elif (time.time() - silence_start) * 1000 >= SIL_MS:
                break
        if (time.time() - start) * 1000 >= max_ms:
            break
    if not buf:
        return np.zeros((0,), dtype=np.int16)
    return np.concatenate(buf)

def transcribe(pcm16):
    audio = pcm16.astype(np.float32) / 32768.0
    segments, _ = stt_model.transcribe(audio, language="en")
    text = "".join([s.text for s in segments]).strip()
    return text

async def main_loop():
    asyncio.create_task(arbiter.run())
    stream = sd.InputStream(channels=1, samplerate=SR, dtype='int16',
                            callback=audio_callback, blocksize=frame_len)
    stream.start()
    print("Buddy Windows listener started. Say the wake phrase when you're ready.")

    while True:
        # short window: listen for candidate speech
        pcm = record_until_silence(max_ms=2000)
        if pcm.size == 0:
            await asyncio.sleep(0.05)
            continue

        # Wake gate
        if not heard_wake(pcm):
            continue

        if not arbiter.has_token:
            await asyncio.sleep(0.3)
            continue

        await tts_say("I'm listening.")
        utt = record_until_silence()
        if utt.size == 0:
            continue

        text = transcribe(utt)
        if not text:
            continue

        try:
            r = requests.post(SERVER_URL, json={"text": text, "device": DEVICE}, timeout=30)
            r.raise_for_status()
            reply = r.json().get("reply_text", "I'm here.")
        except Exception:
            reply = "I couldn't reach the local brain."

        await tts_say(reply)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass
