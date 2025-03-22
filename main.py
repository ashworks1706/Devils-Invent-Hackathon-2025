
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

from google import genai
import re
import json
from dobot_controller import DobotController

# Initialize the robot
robot = DobotController()

# Return home
robot.home()

# Move to a position

# system_instruction=  """

# You are an AI assistant controlling a robotic arm. You can help the user by moving the arm and manipulating objects.
# You have these capabilities:
# - move_to(x,y,z): Move the arm to x,y,z coordinates (default 1)
# - grab(): Pick up an object at the current position
# - drop(): Release the currently held object
# - move_relative(x,y,z): Move the arm relative to its current position
# - home() : Return the arm to its home position
# - set_movement_speed(velocity, acceleration): Set the movement speed and acceleration for the arm

# When asked to perform physical tasks, use these functions by stating the command clearly.
# For example, say "I'll move the arm to the left" or "Let me grab that object for you".
# Always confirm actions by describing what you're doing.
# """



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
You are an AI assistant controlling a robotic arm. You can help the user by moving the arm and manipulating objects.
You have these capabilities:
- move_to(x,y,z): Move the arm to x,y,z coordinates (default 1)
- grab(): Pick up an object at the current position
- drop(): Release the currently held object
- move_relative(x,y,z): Move the arm relative to its current position
- home() : Return the arm to its home position
- set_movement_speed(velocity, acceleration): Set the movement speed and acceleration for the arm

When asked to perform physical tasks, use these functions by stating the command clearly.
For example, say "I'll move the arm to the left" or "Let me grab that object for you".
Always confirm actions by describing what you're doing.

RESPONSE ONLY IN JSON FORMAT

FOLLOW THIS GUIDELINE FOR THE JSON FORMAT. ALL ARGUMENTS ARE REQUIRED.

1. move_to function :
```json
{
    "function": "move_to",
    "arguments": {
        "x": 10,
        "y": 20,
        "z": 30
    }
}
```
2. move_relative function :
```json
{
    "function": "move_relative",
    "arguments": {
        "x": 10,
        "y": 20,
        "z": 30
    }
}
```

"""
# 3. grab function :
# ``` json
# {
#     "function": "grab",
#     "arguments": {}
# }
# ```
# 4. drop function :
# ```json
# {
#     "function": "drop",
#     "arguments": {}
# }
# ```
# 5. home function :

# {
#     "function": "home",
#     "arguments": {}
# }
# 6. set_movement_speed function :
# {
#     "function": "set_movement_speed",
#     "arguments": {
#         "velocity": 10,
#         "acceleration": 10
#     }
# }

# Update the CONFIG to include the system instruction


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
        
        # Movement functions and state with 3D coordinates
        self.current_position = {"x": 0, "y": 0, "z": 0}
        self.holding_object = False
        print("Robot initialized with position:", self.current_position)

    def home(self):
        """Return the robot to its home position"""
        print("Returning to home position")
        # robot.home()
        self.current_position = {"x": 0, "y": 0, "z": 0}
        return "Returned to home position"
    
    def set_movement_speed(self, velocity=10, acceleration=10):
        """Set the movement speed and acceleration for point-to-point movements."""
        print(f"Setting movement speed: velocity={velocity}, acceleration={acceleration}")
        # robot.set_movement_speed(velocity=velocity, acceleration=acceleration)
        return f"Movement speed set to velocity={velocity}, acceleration={acceleration}"
    # Movement functions
    
    def move_to(self,x : float, y: float, z : float):
        """Move to absolute coordinates"""
            
        print(f"Moving to position: x={x}, y={y}, z={z}")
        # Execute actual robot movement
        robot.move_to(x, y, z)
        return f"Moving to position: x={x}, y={y}, z={z}"
        
    def move_relative(self,x : float, y :float, z :float):
        """Move the robot relative to its current position."""
                
        print(f"Moving relatively: x={x}, y={y}, z={z}")
        
        # Execute actual robot movement
        robot.move_relative(x, y, z)
        return f"Moved relatively by ({x}, {y}, {z}) to position ({self.current_position['x']}, {self.current_position['y']}, {self.current_position['z']})"
        
    def grab(self):
        """Grab object at current or specified position"""
        # Move to position if specified
            
        print("Grabbing object")
        robot.set_gripper(enable=True, grip=True)
        return "Object grabbed successfully"
            
    def drop(self):
        """Drop held object at current or specified position"""
        
        print("Dropping object")
        robot.set_gripper(enable=True, grip=False)
        return "Object dropped successfully"

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
                    # cv2.VideoCapture, "/dev/video0"
                    # integrated camera
                    # external camera
                    cv2.VideoCapture, "/dev/video4"
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

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

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

    def process_tool_calls(self, response):
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
                print("balls")
                print(response_text)
                print("balls")
                tool_call = json.loads(response_text)  # Convert JSON string to dictionary
                function_name = tool_call.get("function")
                function_args = tool_call.get("arguments", {})
                print(type(function_args))
                print(f"Debug: Function called: {function_name} with args: {function_args}")
                
                # Execute the appropriate function based on the tool call
                if function_name == "move_to":
                    print("executing move_to",function_args)
                    result = self.move_to(function_args["x"], function_args["y"], function_args["z"])
                elif function_name == "move_relative":
                    print("executing move_relative",function_args)
                    result = self.move_relative(function_args["x"], function_args["y"], function_args["z"])
                elif function_name == "grab":
                    result = self.grab()
                elif function_name == "drop":
                    result = self.drop()
                elif function_name == "home":
                    result = self.home()
                elif function_name == "set_movement_speed":
                    result = self.set_movement_speed(**function_args)
                else:
                    result = f"Unknown function: {function_name}"
                
                print(f"Debug: Function result: {result}")
                return result
            except json.JSONDecodeError as json_e:
                print(f"Debug: Error parsing response text as JSON: {json_e}")
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
                tool_result = self.process_tool_calls(response)
                # if tool_result:
                #     print(f"Tool execution result: {tool_result}")
                
                # if data := response.data:
                #     self.audio_in_queue.put_nowait(data)
                #     continue
                
                # if text := response.text:
                #     print("Text received!")

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
                        self.move_to, self.move_relative
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
    
    
