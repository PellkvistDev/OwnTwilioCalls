import os
import base64
import io
import asyncio
import webrtcvad
import audioop

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect
import openai

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

# Get OpenAI API key from environment variable
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Eller ersätt med din nyckel direkt

vad = webrtcvad.Vad(1)  # 0-3, 1 is low aggressiveness

def transcribe_pcm(pcm_audio_bytes):
    audio_file = io.BytesIO(pcm_audio_bytes)
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file, 
        response_format="text"
    )        
    return transcription.text

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

                # Convert mulaw 8-bit to PCM16 LE 16-bit mono (required for VAD)
                pcm16 = audioop.ulaw2lin(mulaw_bytes, 2)

                # Run VAD on frame
                is_speech = vad.is_speech(pcm16, sample_rate=8000)

                if is_speech:
                    buffer.extend(pcm16)
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

                # Echo back audio (optional)
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
