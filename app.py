import os
import base64
import io
import asyncio
import webrtcvad
import wave

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect
import openai
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.api_route("/voice", methods=["GET", "POST"])
async def voice(request: Request):
    print("Call executed")
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url="wss://owntwiliocalls.onrender.com/media")
    response.append(connect)
    response.say(
        'This TwiML instruction is unreachable unless the Stream is ended by your WebSocket server.'
    )
    return Response(content=str(response), media_type="application/xml")

# OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

vad = webrtcvad.Vad(1)  # 0-3, 1 is low aggressiveness


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate=8000, sample_width=2, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    buffer.seek(0)
    return buffer


def transcribe_pcm(pcm_audio_bytes):
    wav_file = pcm_to_wav_bytes(pcm_audio_bytes)

    # OpenAI API expects a named file-like object
    with io.BytesIO(wav_file.read()) as f:
        f.name = "audio.wav"  # Required: must have a .wav extension
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    return transcript



def mulaw_to_pcm16(mulaw_bytes):
    # Mu-law to PCM16 conversion without audioop (manual table)
    import soundfile as sf
    import tempfile
    import subprocess

    # Temporary workaround using ffmpeg for conversion (you can replace this later)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ulaw") as f:
        f.write(mulaw_bytes)
        f.flush()
        out_wav = f.name.replace(".ulaw", ".wav")
        subprocess.run([
            "ffmpeg", "-y", "-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", f.name, out_wav
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out_wav, "rb") as out_f:
            data = out_f.read()
    return data


@app.websocket("/media")
async def media_ws(websocket: WebSocket):
    print("Websocket tries to connect")
    await websocket.accept()
    print("🔌 WebSocket connected.")

    buffer = bytearray()
    silence_frames = 0
    max_silence_frames = 10  # ~200ms silence at 20ms frames

    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("event")

            if event == "start":
                print("🚀 Call started.")
                buffer.clear()
                silence_frames = 0

            elif event == "media":
                payload = data["media"]["payload"]
                mulaw_bytes = base64.b64decode(payload)

                # Convert to PCM16 for VAD
                pcm16 = np.frombuffer(mulaw_bytes, dtype=np.uint8).astype(np.int16)
                pcm_bytes = pcm16.tobytes()

                is_speech = vad.is_speech(pcm_bytes, sample_rate=8000)

                if is_speech:
                    buffer.extend(pcm_bytes)
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames > max_silence_frames and len(buffer) > 0:
                        print("Silence detected, transcribing chunk...")
                        loop = asyncio.get_event_loop()
                        transcript = await loop.run_in_executor(None, transcribe_pcm, bytes(buffer))
                        print("Transcription:", transcript)
                        buffer.clear()
                        silence_frames = 0

                await websocket.send_json({
                    "event": "media",
                    "media": {
                        "payload": payload
                    }
                })

            elif event == "stop":
                if len(buffer) > 0:
                    print("Call ended, transcribing last chunk...")
                    loop = asyncio.get_event_loop()
                    transcript = await loop.run_in_executor(None, transcribe_pcm, bytes(buffer))
                    print("Transcription:", transcript)
                    buffer.clear()

                print("🛑 Call ended.")
                break

    except Exception as e:
        print(f"⚠️ Error: {e}")

    finally:
        await websocket.close()
        print("🔒 WebSocket closed.")
