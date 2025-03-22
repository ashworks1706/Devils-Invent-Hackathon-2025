import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import asyncio
from google import genai as genai_vertex
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

def move_forward(x: int):
    """Execute forward movement by x units"""
    print(f"🚀 Moving forward by {x} units!")
    engine.say(f"Moving forward by {x} units")
    engine.runAndWait()

def move_backward(x: int):
    """Execute backward movement by x units"""
    print(f"🚀 Moving backward by {x} units!")
    engine.say(f"Moving backward by {x} units")
    engine.runAndWait()

def move_up(x: int):
    """Execute upward movement by x units"""
    print(f"🚀 Moving up by {x} units!")
    engine.say(f"Moving up by {x} units")
    engine.runAndWait()

def move_down(x: int):
    """Execute downward movement by x units"""
    print(f"🚀 Moving down by {x} units!")
    engine.say(f"Moving down by {x} units")
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
                ),
                genai.protos.FunctionDeclaration(
                    name="move_forward",
                    description="Moves object forward by a specified number of units",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "x": content.Schema(
                                type=content.Type.INTEGER,
                                description="Number of units to move forward"
                            )
                        },
                        required=["x"]
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name="move_backward",
                    description="Moves object backward by a specified number of units",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "x": content.Schema(
                                type=content.Type.INTEGER,
                                description="Number of units to move backward"
                            )
                        },
                        required=["x"]
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name="move_up",
                    description="Moves object upward by a specified number of units",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "x": content.Schema(
                                type=content.Type.INTEGER,
                                description="Number of units to move up"
                            )
                        },
                        required=["x"]
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name="move_down",
                    description="Moves object downward by a specified number of units",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "x": content.Schema(
                                type=content.Type.INTEGER,
                                description="Number of units to move down"
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
        response = await chat.send_message_async(command)
        print("raw response : ", response)
        for part in response.parts:
            if hasattr(part, 'function_call') and part.function_call:
                func_call = part.function_call
                if func_call.name == "move_left":
                    move_left(**func_call.args)
                elif func_call.name == "move_right":
                    move_right(**func_call.args)
                elif func_call.name == "move_forward":
                    move_forward(**func_call.args)
                elif func_call.name == "move_backward":
                    move_backward(**func_call.args)
                elif func_call.name == "move_up":
                    move_up(**func_call.args)
                elif func_call.name == "move_down":
                    move_down(**func_call.args)
            elif hasattr(part, 'text'):
                print(f"🤖: {part.text}")
                engine.say(part.text)
                engine.runAndWait()
                
    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        engine.say("Sorry, I encountered an error processing that request")
        engine.runAndWait()


async def get_voice_input():
    """Capture voice input from microphone"""
    print("🎤 Capturing voice input...")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        r.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(None, r.listen, source, 5)
        try:
            text = await loop.run_in_executor(None, r.recognize_google, audio)
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
        command = (await get_voice_input()).lower()
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
