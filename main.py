

# Add this import at the top
import numpy as np
import asyncio
import base64
import io
import os
import sys
import traceback
import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai
import re
from dobot_controller import DobotController

# Initialize the robot
robot = DobotController()

# Return home
robot.home()

# Move to a position




FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"

# os.environ["QT_QPA_PLATFORM"] = "wayland"

DEFAULT_MODE = "camera"

client = genai.Client(api_key='AIzaSyAmE4CWdWDsaea26wCC5lSYGYzAosWDi0I',http_options={"api_version": "v1alpha"})

# While Gemini 2.0 Flash is in experimental preview mode, only one of AUDIO or
# TEXT may be passed here.
SYSTEM_INSTRUCTION = """
You are an AI assistant controlling a robotic arm. You can help the user by moving the arm and manipulating objects.
You have these capabilities:
- move_to(x,y,z): Move the arm to x,y,z coordinates (default 1)
- grab(): Pick up an object at the current position
- drop(): Release the currently held object
- move_relative(dx,dy,dz): Move the arm relative to its current position
- home() : Return the arm to its home position
- set_movement_speed(velocity, acceleration): Set the movement speed and acceleration for the arm

When asked to perform physical tasks, use these functions by stating the command clearly.
For example, say "I'll move the arm to the left" or "Let me grab that object for you".
Always confirm actions by describing what you're doing.
"""

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
        robot.home()
        self.current_position = {"x": 0, "y": 0, "z": 0}
        return "Returned to home position"
    
    def set_movement_speed(self, velocity=10, acceleration=10):
        """Set the movement speed and acceleration for point-to-point movements."""
        print(f"Setting movement speed: velocity={velocity}, acceleration={acceleration}")
        robot.set_movement_speed(velocity=velocity, acceleration=acceleration)
        return f"Movement speed set to velocity={velocity}, acceleration={acceleration}"
    # Movement functions
    
    def move_to(self, x=None, y=None, z=None):
        """Move to absolute coordinates"""
        # Update only the provided coordinates
        if x is not None:
            self.current_position["x"] = x
        if y is not None:
            self.current_position["y"] = y
        if z is not None:
            self.current_position["z"] = z
            
        print(f"Moving to position: x={self.current_position['x']}, y={self.current_position['y']}, z={self.current_position['z']}")
        # Execute actual robot movement
        robot.move_to(self.current_position["x"], self.current_position["y"], self.current_position["z"])
        return f"Moved to position ({self.current_position['x']}, {self.current_position['y']}, {self.current_position['z']})"
        
    def move_relative(self, dx=0, dy=0, dz=0):
        """Move the robot relative to its current position."""
        # Calculate new position
        self.current_position["x"] += dx
        self.current_position["y"] += dy
        self.current_position["z"] += dz
        
        print(f"Moving relatively: dx={dx}, dy={dy}, dz={dz}")
        print(f"New position: x={self.current_position['x']}, y={self.current_position['y']}, z={self.current_position['z']}")
        
        # Execute actual robot movement
        robot.move_relative(dx=dx, dy=dy, dz=dz)
        return f"Moved relatively by ({dx}, {dy}, {dz}) to position ({self.current_position['x']}, {self.current_position['y']}, {self.current_position['z']})"
        
    def grab(self, x=None, y=None, z=None):
        """Grab object at current or specified position"""
        # Move to position if specified
        if x is not None or y is not None or z is not None:
            self.move_to(x, y, z)
            
        print("Grabbing object")
        robot.set_gripper(enable=True, grip=True)
        self.holding_object = True
        return "Object grabbed successfully"
            
    def drop(self, x=None, y=None, z=None):
        """Drop held object at current or specified position"""
        # Move to position if specified
        if x is not None or y is not None or z is not None:
            self.move_to(x, y, z)
            
        if not self.holding_object:
            print("No object currently held")
            return "No object to drop"
            
        print("Dropping object")
        robot.set_gripper(enable=True, grip=False)
        self.holding_object = False
        return "Object dropped successfully"

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

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
                    cv2.VideoCapture, 0
                )  # 0 represents the default camera
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

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                print("RAW RESPONSE : ",text)
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    # Process commands in the text response
                    processed_text = self.process_commands(text)
                    # print(processed_text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()
    
    def process_commands(self, text):
        """Process text for movement commands and execute them"""
        # Define command patterns to look for
        commands = {
            "move to": self.move_to,
            "grab": self.grab,
            "drop": self.drop,
            "move relative": self.move_relative,
            "home": self.home,
            "set movement speed": self.set_movement_speed
        }
        
        result_text = text
        
        
        
        # Check for commands with coordinates
        for cmd, func in commands.items():
            if cmd == "set movement speed":
                # Special pattern for set_movement_speed
                pattern = r"set movement speed(?:[:\s]+velocity=(\d+)(?:,?\s+acceleration=(\d+))?|[:\s]+(\d+)(?:,?\s+(\d+))?)"
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    velocity = match.group(1) or match.group(3)
                    acceleration = match.group(2) or match.group(4)
                    
                    velocity = int(velocity) if velocity else 10
                    acceleration = int(acceleration) if acceleration else 10
                    
                    response = func(velocity, acceleration)
                    result_text = result_text.replace(match.group(0), f"[{response}]", 1)
            elif cmd in ["move to", "move relative"]:
                # Pattern for 3D coordinates
                pattern = fr"{cmd}(?:\s+to|\s+by)?(?:\s+coordinates?\s*\((-?\d+(?:\.\d+)?)(?:,?\s*)(-?\d+(?:\.\d+)?)(?:,?\s*)(-?\d+(?:\.\d+)?)?\)|\s+x=(-?\d+(?:\.\d+)?)(?:,?\s*)y=(-?\d+(?:\.\d+)?)(?:,?\s*)z=(-?\d+(?:\.\d+)?)?)"
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract coordinates from either group format (1,2,3) or (4,5,6)
                    x = float(match.group(1) or match.group(4)) if (match.group(1) or match.group(4)) else None
                    y = float(match.group(2) or match.group(5)) if (match.group(2) or match.group(5)) else None
                    z = float(match.group(3) or match.group(6)) if (match.group(3) or match.group(6)) else None
                    
                    response = func(x, y, z)
                    result_text = result_text.replace(match.group(0), f"[{response}]", 1)
            elif cmd in ["grab", "drop"]:
                # Pattern for grab/drop with optional coordinates
                pattern = fr"{cmd}(?:\s+at\s+coordinates?\s*\((-?\d+(?:\.\d+)?)(?:,?\s*)(-?\d+(?:\.\d+)?)(?:,?\s*)(-?\d+(?:\.\d+)?)?\)|\s+at\s+x=(-?\d+(?:\.\d+)?)(?:,?\s*)y=(-?\d+(?:\.\d+)?)(?:,?\s*)z=(-?\d+(?:\.\d+)?)?)"
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    x = float(match.group(1) or match.group(4)) if (match.group(1) or match.group(4)) else None
                    y = float(match.group(2) or match.group(5)) if (match.group(2) or match.group(5)) else None
                    z = float(match.group(3) or match.group(6)) if (match.group(3) or match.group(6)) else None
                    
                    response = func(x, y, z)
                    result_text = result_text.replace(match.group(0), f"[{response}]", 1)
        
        # Check for simple commands without coordinates
        simple_patterns = {
            "home": r"\bhome\b",
            "grab": r"\bgrab\b",
            "drop": r"\bdrop\b"
        }
        
        for cmd, pattern in simple_patterns.items():
            if cmd in commands and re.search(pattern, text, re.IGNORECASE):
                # Make sure it's not already replaced
                if f"[{commands[cmd].__name__}" not in result_text:
                    response = commands[cmd]()
                    # Add the response to the text
                    result_text += f"\n[{response}]"
                    
        return result_text


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

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config={"response_modalities": ["TEXT"],"system_instruction": SYSTEM_INSTRUCTION
                                                             
                                                            #  ,tools : [set_light_values]
                                                             }
                                        ) as session, asyncio.TaskGroup() as tg,):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

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
    
    
