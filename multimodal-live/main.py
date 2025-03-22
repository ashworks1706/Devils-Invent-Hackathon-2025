"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```
"""

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


FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"

# os.environ["QT_QPA_PLATFORM"] = "wayland"

DEFAULT_MODE = "camera"

client = genai.Client(api_key='',http_options={"api_version": "v1alpha"})

# While Gemini 2.0 Flash is in experimental preview mode, only one of AUDIO or
# TEXT may be passed here.
SYSTEM_INSTRUCTION = """
You are an AI assistant controlling a robotic arm. You can help the user by moving the arm and manipulating objects.
You have these capabilities:
- move_left(x): Move the arm left by x units (default 1)
- move_right(x): Move the arm right by x units (default 1)
- move_forward(y): Move the arm forward by y units (default 1)
- move_backward(y): Move the arm backward by y units (default 1)
- grab(): Pick up an object at the current position
- drop(): Release the currently held object

When asked to perform physical tasks, use these functions by stating the command clearly.
For example, say "I'll move the arm to the left" or "Let me grab that object for you".
Always confirm actions by describing what you're doing.
"""

# Update the CONFIG to include the system instruction
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": SYSTEM_INSTRUCTION
}


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
        
        # Movement functions and state
        self.current_position = {"x": 0, "y": 0}
        self.holding_object = False

    # Movement functions
    def move_left(self, x=None, y=None):
        """Move left by specified amount or default step"""
        amount = x if x is not None else 1
        self.current_position["x"] -= amount
        print(f"Moving left to position: {self.current_position}")
        return f"Moved left to position {self.current_position}"
        
    def move_right(self, x=None, y=None):
        """Move right by specified amount or default step"""
        amount = x if x is not None else 1
        self.current_position["x"] += amount
        print(f"Moving right to position: {self.current_position}")
        return f"Moved right to position {self.current_position}"
        
    def move_forward(self, x=None, y=None):
        """Move forward by specified amount or default step"""
        amount = y if y is not None else 1
        self.current_position["y"] += amount
        print(f"Moving forward to position: {self.current_position}")
        return f"Moved forward to position {self.current_position}"
        
    def move_backward(self, x=None, y=None):
        """Move backward by specified amount or default step"""
        amount = y if y is not None else 1
        self.current_position["y"] -= amount
        print(f"Moving backward to position: {self.current_position}")
        return f"Moved backward to position {self.current_position}"
        
    def grab(self, x=None, y=None):
        """Grab object at current or specified position"""
        position = {"x": x if x is not None else self.current_position["x"], 
                   "y": y if y is not None else self.current_position["y"]}
        if not self.holding_object:
            self.holding_object = True
            print(f"Grabbing object at position: {position}")
            return f"Grabbed object at position {position}"
        else:
            print("Already holding an object")
            return "Already holding an object"
            
    def drop(self, x=None, y=None):
        """Drop held object at current or specified position"""
        position = {"x": x if x is not None else self.current_position["x"], 
                   "y": y if y is not None else self.current_position["y"]}
        if self.holding_object:
            self.holding_object = False
            print(f"Dropping object at position: {position}")
            return f"Dropped object at position {position}"
        else:
            print("No object to drop")
            return "No object to drop"

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
                    print("Frame read")
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
                    print("Frame displayed")
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
                    print("Frame processed")
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
                
                try:
                    # Check for key press (q to quit)
                    key = cv2.waitKey(1) & 0xFF
                    print("Checking for key press")
                    if key == ord('q'):
                        print("User requested exit")
                        break
                except Exception as e:
                    print(f"Error checking key press: {e}")

                try:
                    await asyncio.sleep(1.0)
                    await self.out_queue.put({"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()})
                    print("Frame sent to model")
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
                print("Camera resources released")
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
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    # Process commands in the text response
                    processed_text = self.process_commands(text)
                    print(processed_text, end="")

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
            "move left": self.move_left,
            "move right": self.move_right,
            "move forward": self.move_forward,
            "move backward": self.move_backward,
            "grab": self.grab,
            "drop": self.drop
        }
        
        result_text = text
        
        # Check for commands with coordinates
        for cmd, func in commands.items():
            # Pattern: command with optional coordinates
            pattern = fr"{cmd}(?: to coordinates? ?\((-?\d+)(?:,|\s+)?\s*(-?\d+)\)| by ?\((-?\d+)(?:,|\s+)?\s*(-?\d+)\)| to x=(-?\d+)[ ,]*y=(-?\d+)| by x=(-?\d+)[ ,]*y=(-?\d+))?"
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract coordinates from any of the possible capture groups
                x = y = None
                for i in range(1, 9, 2):  # Check odd indices for x values
                    if match.group(i) and match.group(i).isdigit():
                        x = int(match.group(i))
                        break
                        
                for i in range(2, 9, 2):  # Check even indices for y values
                    if match.group(i) and match.group(i).isdigit():
                        y = int(match.group(i))
                        break
                
                # Execute the command
                response = func(x, y)
                # Replace the command in the text with the response
                result_text = result_text.replace(match.group(0), f"[{response}]", 1)
        
        # Check for simple commands without coordinates
        for cmd, func in commands.items():
            if cmd in text.lower() and cmd not in result_text.lower():
                response = func()
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
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
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