from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream


app = FastAPI()

# Allow any CORS for testing
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.api_route("/voice", methods=["GET", "POST"])
async def voice(request: Request):
    """Respond with TwiML to start a bidirectional media stream."""
    print("Call executed")
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url="wss://owntwiliocalls.onrender.com/media")
    response.append(connect)
    response.say('The stream has started.')
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/media")
async def media_ws(websocket: WebSocket):
    print("Websocket tries to connect")
    await websocket.accept()
    print("🔌 WebSocket connected.")

    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("event")

            if event == "start":
                print("🚀 Call started.")
            elif event == "media":
                payload = data["media"]["payload"]

                # Echo the audio back to Twilio
                await websocket.send_json({
                    "event": "media",
                    "media": {
                        "payload": payload
                    }
                })

            elif event == "stop":
                print("🛑 Call ended.")
                break

    except Exception as e:
        print(f"⚠️ Error: {e}")
    finally:
        await websocket.close()
        print("🔒 WebSocket closed.")
