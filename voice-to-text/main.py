import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import asyncio
from google import genai as genai_vertex
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
import speech_recognition as sr
import pyttsx3
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Movement functions
def move_left(x: int):
    """Execute left movement by x units"""
    print(f"🚀 Moving left by {x} units!")
    engine.say(f"Moving left by {x} units")
    engine.runAndWait()

def move_right(x: int):
    """Execute right movement by x units"""
    print(f"🚀 Moving right by {x} units!")
    engine.say(f"Moving right by {x} units")
    engine.runAndWait()

# Configure Gemini model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={
        "temperature": 0.0,
        "top_p": 0.1,
        "max_output_tokens": 100,
    },
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    },
    tools=[
        genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="move_left",
                    description="Moves object to the left by a specified number of units",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "x": content.Schema(
                                type=content.Type.INTEGER,
                                description="Number of units to move left"
                            )
                        },
                        required=["x"]
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name="move_right",
                    description="Moves object to the right by a specified number of units",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "x": content.Schema(
                                type=content.Type.INTEGER,
                                description="Number of units to move right"
                            )
                        },
                        required=["x"]
                    )
                )
            ]
        )
    ],
    tool_config={'function_calling_config': 'ANY'}
)
chat = model.start_chat(history=[])

async def process_voice_command(command: str):
    """Process voice command through Gemini"""
    try:
        print("hello")
        response = await chat.send_message_async(command)
        print("raw response : ", response)
        for part in response.parts:
            if hasattr(part, 'function_call') and part.function_call:
                func_call = part.function_call
                if func_call.name == "move_left":
                    move_left(**func_call.args)
                elif func_call.name == "move_right":
                    move_right(**func_call.args)
            elif hasattr(part, 'text'):
                print(f"🤖: {part.text}")
                engine.say(part.text)
                engine.runAndWait()
                
    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        engine.say("Sorry, I encountered an error processing that request")
        engine.runAndWait()


def get_voice_input():
    """Capture voice input from microphone"""
    print("🎤 Capturing voice input...")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = r.listen(source, timeout=5)
        try:
            text = r.recognize_google(audio)
            print(f"🎤 You said: {text}")
            return text
        except sr.UnknownValueError:
            print("🚨 Could not understand audio")
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
        print(f"🔍 Received command: {command}")
        
        if not command:
            print("⚠️ No command detected, retrying...")
            continue
            
        if "exit" in command:
            print("🛑 Shutting down...")
            break
        
        await process_voice_command(command)

if __name__ == "__main__":
    try:
        print("🚀 Starting Voice Agent...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Program terminated by user")
