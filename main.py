
# sudo chmod 666 /dev/ttyACM0

# Add this import at the top
import numpy as np
import asyncio
from google.genai.types import Content 
import base64
import io
import os
import sys
import traceback
import cv2
import pyaudio
from google.genai import types
import PIL.Image
import mss

import argparse
import pyttsx3

from google import genai
import re
import json
# from dobot_controller import DobotController

# # Initialize the robot
# robot = DobotController()

# # Return home
# robot.home()

MAX_Z = 200
MIN_Z = -200



FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"

# os.environ["QT_QPA_PLATFORM"] = "wayland"

DEFAULT_MODE = "camera"

client = genai.Client(api_key='AIzaSyAR_4Nk8x9jq2rl4FIZ6v4OudZSuwvYyDg',http_options={"api_version": "v1alpha"})

# While Gemini 2.0 Flash is in experimental preview mode, only one of AUDIO or
# TEXT may be passed here.
SYSTEM_INSTRUCTION = """
You are an AI assistant controlling a robotic arm. You can help the user by moving the arm and manipulating objects. You have to follow the grid on the background to identify where to move. The scripts are hard coded so you just have to identify the object and use your function for relative position from the board.
You have these capabilities:
- pickup_from_to(pickup_pos, dropoff_pos): Move the arm to pickup position coordinate (eg. A5, D5, etc), grab object and drop at dropoff position coordinate (eg. A5, D5, etc).
- home(): Return the robot to its home position and cancel any operations.

When asked to perform physical tasks, use these functions by stating the command clearly.

RESPONSE ONLY IN JSON FORMAT WHEN REQUIRED

FOLLOW THIS GUIDELINE FOR THE JSON FORMAT. ALL ARGUMENTS ARE REQUIRED.

1. pickup_from_to function :
```json
{
    "function": "pickup_from_to",
    "arguments": {
        "pickup_pos": "D5",
        "dropoff_pos": "D2"
    }
}
```
2. home function :
```json
{
    "function": "home",
    "arguments": {}
}
```

"""

pya = pyaudio.PyAudio()


class AudioLoop:
    
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.coordinates={}
        self.holding_object = False

    async def home(self):
        """Return the robot to its home position"""
        await self.speak_text("Cancelling positions and Returning to home position")
        # robot.home()
        return "Cancelled operations and Returned to home position"
    
    
    async def pickup_from_to(self, pickup_pos: str, dropoff_pos: str):
        """Move to relative pos coordinates and pickup then drop to position"""
            
        await self.speak_text(f"Moving on to {pickup_pos} for pickup and dropped to {dropoff_pos}")
        
        # # map pos coordinate
        
        # x,y,z = self.coordinates[pos]
        
        # self.speak_text(f"Moving on X axis: x={x}, y={y} for pickup")
                
        # robot.move_to(x, y, MAX_Z)
        
        # self.speak_text(f"Going down on z axis: z={z}")
                
        # robot.move_to(x, y, MIN_Z)
        
        # self.speak_text("Grabbing object")
        
        # robot.set_gripper(enable=True, grip=True)
        
        # self.speak_text(f"Going up on z axis: z={z}")
                
        # robot.move_to(x, y, MAX_Z)
        
        # self.speak_text(f"Moving on X axis: x={x}, y={y} for dropoff")
                
        # self.speak_text(f"Going down on z axis: z={z} for dropoff ")
                
        # robot.move_to(x, y, MIN_Z)
        
        # self.speak_text("Dropping object")
        
        # robot.set_gripper(enable=True, grip=False)
        
        
        
        return f"Picked up object from position : {pickup_pos} and  dropped the object to position: {dropoff_pos}"
        
        

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input= f"User Prompt : {text}", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        try:
            # This takes about a second, and will block the whole program
            # causing the audio pipeline to overflow if you don't to_thread it.
            try:
                cap = await asyncio.to_thread(
                    cv2.VideoCapture, "/dev/video0"
                    # integrated camera
                    # external camera
                    # cv2.VideoCapture, "/dev/video4"
                )  
                print("Camera initialized successfully")
            except Exception as e:
                print(f"Error initializing camera: {e}")
                return
            
            # Create a window to display the video
            try:
                cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
                print("Display window created")
            except Exception as e:
                print(f"Error creating window: {e}")
                cap.release()
                return

            while True:
                try:
                    # Get frame for processing
                    ret, frame = await asyncio.to_thread(cap.read)
                    if not ret:
                        print("Failed to capture frame")
                        break
                except Exception as e:
                    print(f"Error reading frame: {e}")
                    break
                    
                try:
                    # Display the frame
                    # await asyncio.to_thread(cv2.imshow, "Camera Feed", frame)
                    cv2.imshow( "Camera Feed", frame)
                except Exception as e:
                    print(f"Error displaying frame: {e}")
                
                try:
                    # Process frame for sending to model
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = PIL.Image.fromarray(frame_rgb)
                    img.thumbnail([1024, 1024])

                    image_io = io.BytesIO()
                    img.save(image_io, format="jpeg")
                    image_io.seek(0)

                    mime_type = "image/jpeg"
                    image_bytes = image_io.read()
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
                
                try:
                    # Check for key press (q to quit)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User requested exit")
                        break
                except Exception as e:
                    print(f"Error checking key press: {e}")

                try:
                    await asyncio.sleep(1.0)
                    await self.out_queue.put({"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()})
                except Exception as e:
                    print(f"Error sending frame to model: {e}")
                
        except Exception as e:
            print(f"Unexpected error in get_frames: {e}")
            traceback.print_exc()
        finally:
            try:
                # Release the VideoCapture object and close windows
                if 'cap' in locals():
                    cap.release()
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error releasing resources: {e}")

    async def get_screen(self):
        # Create a window to display the screen capture
        cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
        
        while True:
            # Get screen capture
            sct = mss.mss()
            monitor = sct.monitors[0]
            screenshot = await asyncio.to_thread(sct.grab, monitor)
            
            # Convert to numpy array for display
            img_np = np.array(screenshot)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            
            # Display the frame
            await asyncio.to_thread(cv2.imshow, "Screen Capture", img_bgr)
            
            # Process for sending to model
            image_io = io.BytesIO()
            PIL.Image.fromarray(screenshot).save(image_io, format="jpeg")
            image_io.seek(0)

            mime_type = "image/jpeg"
            image_bytes = image_io.read()
            
            # Check for key press (q to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            await asyncio.sleep(1.0)
            await self.out_queue.put({"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()})
        
        cv2.destroyAllWindows()

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        print("Listening for audio...")
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def speak_text(self, text: str):
        """Speak out the given text string."""
        try:
            print(f"Speaking: {text}")
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Adjust speaking rate
            engine.setProperty('volume', 0.9)  # Adjust volume (0.0 to 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error in speak_text: {e}")

    async def process_tool_calls(self, response):
        """Process tool calls from the model response."""
        try:
            if not response.text:
                print("Debug: Response text is None or empty.")
                return None

            print(f"Debug: Processing response text for tool calls... Response text: {response.text}")
            
            # Clean up and normalize the response text
            response_text = response.text.strip().strip('`')  # Remove all backticks
            if response_text.lower().startswith('json'):
                response_text = response_text[4:].strip()  # Remove 'json' prefix

            try:
                tool_call = json.loads(response_text)  # Convert JSON string to dictionary
                function_name = tool_call.get("function")
                function_args = tool_call.get("arguments", {})
                print(type(function_args))
                print(f"Debug: Function called: {function_name} with args: {function_args}")
                
                # Execute the appropriate function based on the tool call
                if function_name == "pickup_from_to":
                    result = await self.pickup_from_to(function_args["pickup_pos"], function_args["dropoff_pos"])
                else:
                    result = f"Unknown function: {function_name}"
                    await self.speak_text(result)
                
                print(f"Debug: Function result: {result}")
                return result
            except json.JSONDecodeError as json_e:
                print(f"Debug: Error parsing response text as JSON: {json_e} for text: {response.text}")
                await self.speak_text(response.text)
        except Exception as e:
            print(f"Error processing response text: {e}")
            return None
        
    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        print("recieveing video")
        while True:
            async for response in self.session.receive():
                print("Response received")  
                # # Check for tool calls
                tool_result = await self.process_tool_calls(response)
                if tool_result:
                    print(f"Tool execution result: {tool_result}")
                
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                
                if text := response.text:
                    print("Text received!")

            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()
                
    async def run(self):
        try:
            print("Starting with tools configuration...")
            
            async with (
                client.aio.live.connect(
                    model=MODEL, 
                        config=types.LiveConnectConfig(
                    response_modalities=["TEXT"],
                            system_instruction=types.Content(parts=[{"text": SYSTEM_INSTRUCTION}]),  
                        tools=[
                        self.pickup_from_to
                        ],
                    )
                ) as session, 
                asyncio.TaskGroup() as tg,
            ):
                print("Session connected successfully with tools configured")
                self.session = session
# 
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Test the tools by sending a message to prompt tool usage
                
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
    
    
