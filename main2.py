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
import time
import argparse
import pyttsx3
from time import sleep
from google import genai
import re
import json
from dobot_controller import DobotController
import speech_recognition as sr


# Initialize the robot
robot = DobotController()

# Return home
robot.home()

MAX_Z = 200
MIN_Z = -20



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
You are an AI assistant EVE controlling a robotic arm. You can help the user by moving the arm and manipulating objects. You have to follow the grid on the background to identify where to move. 


The scripts are hard coded so you just have to identify the object and use your function for relative position from the board.


You have these capabilities:
- pickup_from_to(pickup_pos, dropoff_pos): Move the arm to pickup position coordinate (eg. 1, 20, etc), grab object and drop at dropoff position coordinate (eg. 1, 20, etc).
- home(): Return the robot to its home position and cancel any operations.

When asked to perform physical tasks, use these functions by stating the command clearly.

RESPONSE ONLY IN JSON FORMAT WHEN REQUIRED

FOLLOW THIS GUIDELINE FOR THE JSON FORMAT. ALL ARGUMENTS ARE REQUIRED.

THE PARAMETERS FOR pickup_from_to function are:
pickup_pos: int
dropoff_pos: int


1. pickup_from_to function :
```json
{
    "function": "pickup_from_to",
    "arguments": {
        "pickup_pos": "1",
        "dropoff_pos": "90"
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

DO NOT EXECUTE ANY FUNCTION UNTIL USER

"""

pya = pyaudio.PyAudio()


class AudioLoop:
    
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.receive_audio_task = None
        self.play_audio_task = None
        self.coordinates={}
        self.holding_object = False
        self.recognizer = sr.Recognizer()
        self.is_processing = False

    async def home(self):
        """Return the robot to its home position"""
        await self.speak_text("Cancelling positions and Returning to home position")
        robot.home()
        return "Cancelled operations and Returned to home position"
    
    def get_coordinates(self, grid_index: int):
        """Get the coordinates of the grid"""
        grid_1 = (250, -170)  
        grid_90 = (330, 160) 
        
        # Grid dimensions
        cols = 18
        rows = 5
        
        # Convert to 0-based index
        adjusted_index = grid_index - 1
        
        # Calculate row and column positions
        row = adjusted_index // cols  # 0-4 (5 rows)
        col = adjusted_index % cols   # 0-17 (18 columns)
        
        # Coordinate calculations
        x = grid_1[0] + row * (grid_90[0] - grid_1[0]) / (rows - 1)
        y = grid_1[1] + col * (grid_90[1] - grid_1[1]) / (cols - 1)
        
        return (x, y)
        
    async def pickup_from_to(self, pickup_pos: int, dropoff_pos: int):
        """Move to relative pos coordinates and pickup then drop to position"""
            
        # Return to home position
        await self.speak_text(f"Moving to Home position for recalibration")
        robot.home()
        robot.set_gripper(enable=True, grip=False)
        
        x,y = self.get_coordinates(int(pickup_pos))
        
        await self.speak_text(f"Moving on X axis: x={x}, y={y} for pickup")
        robot.move_to(x, y, MAX_Z, wait=True)
        
        robot.move_to(x, y, MIN_Z, wait=True)
        
        await self.speak_text("Grabbing object")
        
        robot.set_gripper(enable=True, grip=True)
        sleep(2)
        
        robot.home()
        
        x,y = self.get_coordinates(int(dropoff_pos))
        
        await self.speak_text(f"Moving on X axis: x={x}, y={y} for dropoff")
        
        robot.move_to(x, y, MAX_Z, wait=True)
                
        robot.move_to(x, y, MIN_Z + 20, wait=True)
        
        await self.speak_text("Dropping object")
        
        robot.set_gripper(enable=True, grip=False)
        
        robot.home()
        return f"Picked up object from position : {pickup_pos} and  dropped the object to position: {dropoff_pos}"
        
    
    async def get_frames(self):
        try:
            # Initialize the camera
            try:
                cap = await asyncio.to_thread(
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
                    # Add grid overlay with adjustable parameters
                    height, width, _ = frame.shape
                    rows, cols = 5, 18
                    
                    # Define separate zoom factors for x and y axes
                    zoom_factor_x = 0.85  # Zoom level for x axis
                    zoom_factor_y = 2.4  # Zoom level for y axis
                    offset_x_manual = 25 # Manual X offset adjustment
                    offset_y_manual = 0  # Manual Y offset adjustment
                    rotation_angle = 1.8  # Rotation angle in degrees
                    
                    # Calculate the ROI dimensions with separate zoom factors
                    roi_width = int(width / zoom_factor_x)
                    roi_height = int(height / zoom_factor_y)
                    
                    # Calculate offsets to zoom from center
                    offset_x = (width - roi_width) // 2 + offset_x_manual
                    offset_y = (height - roi_height) // 2 + offset_y_manual
                    
                    # Adjust cell dimensions for zoomed area
                    cell_height = roi_height // rows
                    cell_width = roi_width // cols
                    
                    # Calculate rotation center (center of image)
                    rotation_center = (width // 2, height // 2)
                    
                    # Apply rotation if needed
                    if rotation_angle != 0:
                        # Get rotation matrix
                        M = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
                        # Apply rotation to frame
                        frame = cv2.warpAffine(frame, M, (width, height))
                    
                    # Draw grid lines
                    for i in range(rows + 1):
                        y = offset_y + i * cell_height
                        cv2.line(frame, (offset_x, y), (offset_x + roi_width, y), (0, 0, 255), 1)
                    for j in range(cols + 1):
                        x = offset_x + j * cell_width
                        cv2.line(frame, (x, offset_y), (x, offset_y + roi_height), (0, 0, 255), 1)

                    # Add grid numbers
                    for i in range(rows):
                        for j in range(cols):
                            grid_index = (rows - i - 1) * cols + (cols - j)
                            x = offset_x + j * cell_width + 5
                            y = offset_y + i * cell_height + 20
                            cv2.putText(frame, f"{grid_index}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw adjustment parameters on screen for reference
                    cv2.putText(frame, f"Zoom X: {zoom_factor_x:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Zoom Y: {zoom_factor_y:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Offset X: {offset_x_manual}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Offset Y: {offset_y_manual}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Rotation: {rotation_angle}Â°", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # Display the frame
                    cv2.imshow("Camera Feed", frame)
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

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        """Listen for audio and only process commands that start with 'EVE'"""
        print("Listening for keyword 'EVE'...")
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

        # Keep robot at home position when idle
        await self.home()

        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}

        while True:
            try:
                # Convert audio chunk to numpy array for processing
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Convert to audio format that speech recognition can process
                audio = sr.AudioData(data, SEND_SAMPLE_RATE, 2)
                
                # Try to recognize speech
                try:
                    text = await asyncio.to_thread(self.recognizer.recognize_google, audio)
                    text = text.lower()
                    
                    # Check if command starts with "eve"
                    if text.startswith("eve"):
                        print("Keyword 'EVE' detected!")
                        # Extract the command (everything after "eve")
                        command = text[4:].strip()  # Skip "eve" and any following space
                        
                        if command:
                            print(f"Processing command: {command}")
                            # Only send to model if we have a command
                            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
                            await self.speak_text("Processing your command")
                        else:
                            await self.speak_text("Waiting for your command")
                    
                except sr.UnknownValueError:
                    # Speech was unintelligible
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    
            except Exception as e:
                print(f"Error in listen_audio: {e}")
                await asyncio.sleep(0.1)  # Prevent tight loop on error

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
        print("receiving video")
        # Flag to track if a tool is currently executing
        self.is_tool_executing = False
        
        while True:
            async for response in self.session.receive():
                print("Response received")  
                # Check for tool calls
                if not self.is_tool_executing:
                    tool_result = await self.process_tool_calls(response)
                    if tool_result:
                        print(f"Tool execution result: {tool_result}")
                        self.is_tool_executing = True
                        
                        # Pass the function result back to Gemini
                        await self.session.send(input=f"Function called, System generated function result: {tool_result}", end_of_turn=True)
                        
                        # Reset the flag after sending the result
                        self.is_tool_executing = False
                
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
            print("Listening for keyword 'EVE'. Say 'EVE' followed by your command.")
            
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
                self.is_tool_executing = False  # Initialize tool execution flag
# 
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=2)

                # Test the tools by sending a message to prompt tool usage
                send_realtime_task = tg.create_task(self.send_realtime())
                listen_audio_task = tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())

                tg.create_task(self.receive_audio())


                await send_realtime_task
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
        choices=["camera", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())