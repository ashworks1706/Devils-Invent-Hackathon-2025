# voice_agent.py
from google import genai
import os
import speech_recognition as sr
import pyttsx3
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Movement functions
def move_left():
    """Execute left movement"""
    print("🚀 Moving left!")
    # Add your actual movement logic here
    engine.say("Moving left")
    engine.runAndWait()

def move_right():
    """Execute right movement"""
    print("🚀 Moving right!")
    # Add your actual movement logic here
    engine.say("Moving right")
    engine.runAndWait()

# Configure Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={"api_version": "v1alpha"}
)

tools = [
    {
        "name": "move_left",
        "description": "Moves object to the left",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "move_right",
        "description": "Moves object to the right",
        "parameters": {"type": "object", "properties": {}}
    }
]

async def process_voice_command(command: str):
    """Process voice command through Gemini"""
    config = {
        "response_modalities": ["TEXT"],
        "tools": tools,
        "tool_choice": "auto"
    }
    
    async with client.aio.live.connect(
        model="gemini-2.0-flash-exp",
        config=config
    ) as session:
        await session.send(command, end_of_turn=True)
        
        async for response in session.receive():
            if response.tool_calls:
                for call in response.tool_calls:
                    if call.name == "move_left":
                        move_left()
                    elif call.name == "move_right":
                        move_right()
            else:
                print(f"🤖: {response.text}")
                engine.say(response.text)
                engine.runAndWait()

def get_voice_input():
    """Capture voice input from microphone"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = r.listen(source, timeout=5)
        try:
            text = r.recognize_google(audio)
            print(f"🎤 You said: {text}")
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"🚨 Speech recognition error: {e}")
            return ""

async def main():
    """Main execution loop"""
    print("🔊 Voice Agent Activated")
    print("Say 'exit' to quit\n")
    
    while True:
        command = get_voice_input().lower()
        
        if not command:
            continue
            
        if "exit" in command:
            print("🛑 Shutting down...")
            break
            
        await process_voice_command(command)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Program terminated by user")
