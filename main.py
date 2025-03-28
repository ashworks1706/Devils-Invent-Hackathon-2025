import asyncio
import base64
import io
import traceback
import cv2
import pyaudio
from google.genai import types
import PIL.Image
import mss
import time
import argparse
import pyttsx3
from google import genai
import json
from dobot_controller import DobotController
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add file handler for debugging
debug_file = f'multimodal-live/robot_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
file_handler = logging.FileHandler(debug_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)8s | %(message)s'
))
logger.addHandler(file_handler)

# Initialize the robot
robot = DobotController()

# Return home
robot.home()

MAX_Z = 200
MIN_Z = -23
MIN_Z = 0



FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 2048

MODEL = "models/gemini-2.0-flash-exp"

# os.environ["QT_QPA_PLATFORM"] = "wayland"

DEFAULT_MODE = "camera"

client = genai.Client(api_key='AIzaSyAR_4Nk8x9jq2rl4FIZ6v4OudZSuwvYyDg',http_options={"api_version": "v1alpha"})

# While Gemini 2.0 Flash is in experimental preview mode, only one of AUDIO or
# TEXT may be passed here.

SYSTEM_INSTRUCTION = """
You are an AI assistant name "Eeve" controlling a robotic arm. Only listen only when they're talking to you. Your role is to help users manipulate objects using the arm. Follow these guidelines:

1. Use the grid overlay on the camera feed to identify object positions.
2. When picking up objects, estimate the center of mass and use the grid index closest to that point.
3. Only respond with actions when explicitly instructed by the user.
4. Available functions:
    a) pickup_from_to(pickup_block, dropoff_block): Move arm to pickup position, grab object, and move to dropoff position.
    b) pickup_hold(pickup_block): Move arm to pickup position, grab object in arm.
    c) drop_off(dropoff_block): Move arm to dropoff position.
    d) home(): Return robot to home position and cancel operations when required.

5. You can call multiple functions in a single turn. Respond ONLY in JSON format as follows, with a list of function calls:

    ```json
    [
        {
             "function": "pickup_from_to",
             "arguments": {
                  "pickup_block": "", # Required
                  "dropoff_block": "" # Required
             }
        },
        {
             "function": "drop_off",
             "arguments": {
                  "dropoff_block": "" # Required
             }
        }
    ]
    ```

    Each object in the list represents a separate function call.  The entire response must be valid JSON.

    For pickup_from_to:
    ```
    {
         "function": "pickup_from_to",
         "arguments": {
              "pickup_block": "" # Required,
              "dropoff_block": "" # Required,
         }
    }
    ```
    For drop_off:
    ```
    {
         "function": "drop_off",
         "arguments": {
              "dropoff_block": "" # Required
         }
    }
    ```
    For pickup_hold:
    ```
    {
         "function": "pickup_hold",
         "arguments": {
              "pickup_block": "" # Required
         }
    }
    ```

    For home:
    ```
    {
         "function": "home",
         "arguments": {}
    }
    ```
 

7. Grid reference:
    - Bottom left corner: index 18
    - Center: index 45
    - Use appropriate indices for pickup and dropoff positions

8. If the user doesn't request an action, do not initiate any function calls.

9. For non-function responses, use natural language to interact with the user.

Remember: Only perform actions when explicitly instructed. Maintain a helpful and informative tone in all interactions.
"""

pya = pyaudio.PyAudio()



class AudioLoop:
    
    def __init__(self, video_mode=DEFAULT_MODE):
        logger.info("Initializing AudioLoop with video mode: %s", video_mode)
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None
        self.is_tool_executing = False
        self.recieve_response_task = None
        self.play_audio_task = None
        self.coordinates={}
        self.holding_object = False
        # Initialize pyttsx3 engine here
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Adjust speaking rate
        self.engine.setProperty('volume', 0.9)  # Adjust volume (0.0 to 1.0)
        self.task_queue = []  # Initialize task queue


    async def home(self):
        """Return the robot to its home position"""
        logger.info("Returning robot to home position")
        await self.speak_text("Cancelling positions and Returning to home position")
        robot.home()
        logger.debug("Home position reached")
        return "Cancelled operations and Returned to home position. Please proceed with the next action based on the current state"
    
    def get_coordinates(self, grid_index: int):
        """Get the coordinates of the grid"""
        grid_1 = (250, -170)  
        grid_90 = (330, 160) 
        
        
        

        return "Cancelled operations and Returned to home position"
    
    def get_coordinates(self, grid_index: int):
        """Get the coordinates of the grid"""
        grid_1 = (280, -125)
        grid_90 = (300, 220)
        
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
        
    async def pickup_from_to(self, pickup_block: int, dropoff_block: int):
        """Move to relative pos coordinates and pickup to position"""
        task_description = f"Pickup from {pickup_block} to {dropoff_block}"
        self.add_task(task_description)
            
        # Return to home position
        robot.set_gripper(enable=True, grip=False)
        
        x,y = self.get_coordinates(int(pickup_block))
        await self.speak_text(f"Moving to Home position for recalibration")
        robot.home()
        robot.set_gripper(enable=True, grip=False)
        
        await self.speak_text(f"Moving to {pickup_block} for pickup")
        robot.move_to(x, y, MAX_Z, wait=True)
        
        robot.move_to(x, y, MIN_Z, wait=True)
        
        await self.speak_text(f"Moving on X axis: x={x}, y={y} for pickup")
        robot.move_to(x, y, MAX_Z, wait=True)
        
        robot.move_to(x, y, MIN_Z, wait=True)
        
        await self.speak_text("Grabbing object")
        
        robot.set_gripper(enable=True, grip=True)
        time.sleep(2)
        
        robot.home()
        
        x,y = self.get_coordinates(int(dropoff_block))
        
        await self.speak_text(f"Moving on {dropoff_block} for dropoff")
        
        robot.move_to(x, y, MAX_Z, wait=True)
                
        robot.move_to(x, y, MIN_Z + 20, wait=True)
        time.sleep(1)
        robot.home()
        
        x,y = self.get_coordinates(int(dropoff_pos))
        
        await self.speak_text(f"Moving on X axis: x={x}, y={y} for dropoff")
        
        robot.move_to(x, y, MAX_Z, wait=True)
        time.sleep(1)
                
        robot.move_to(x, y, MIN_Z, wait=True)
        
        await self.speak_text("Dropping object")
        
        robot.set_gripper(enable=True, grip=False)
        
        time.sleep(2)
        robot.home()
        return f"Picked up object from position : {pickup_pos} and  dropped the object to position: {dropoff_pos}"
        
        robot.home()
        return f"Tried picking up object from position : {pickup_block} and the object to position: {dropoff_block}. Please confirm if the object is placed correctly in the image. Please proceed with the next action based on the current state"
        
    async def pickup_hold(self, pickup_block: int):
        """Move to relative pos coordinates and pickup to position"""
        task_description = f"Pickup and hold {pickup_block}"
        self.add_task(task_description)
            
        # Return to home position
        robot.set_gripper(enable=True, grip=False)
        
        x,y = self.get_coordinates(int(pickup_block))
        
        await self.speak_text(f"Moving to {pickup_block} for pickup")
        robot.move_to(x, y, MAX_Z, wait=True)
        
        robot.move_to(x, y, MIN_Z, wait=True)
        
        await self.speak_text("Grabbing object")
        
        robot.set_gripper(enable=True, grip=True)
        
        robot.home()
        time.sleep(2)
        
        return f"Tried picking up object from position : {pickup_block} and tried holding the object. Please confirm if the object is placed correctly in the image. Please proceed with the next action based on the current state."
    
    async def drop_off(self, dropoff_block: int):
        """Move to relative pos coordinates and drop to position"""
        task_description = f"Drop off at {dropoff_block}"
        self.add_task(task_description)
            
        
        x,y = self.get_coordinates(int(dropoff_block))
        
        await self.speak_text(f"Moving on {dropoff_block} for dropoff")
        
        robot.move_to(x, y, MAX_Z, wait=True)
                
        robot.move_to(x, y, MIN_Z + 20, wait=True)
        
        await self.speak_text("Dropping object")
        
        robot.set_gripper(enable=True, grip=False)
        
        time.sleep(2)
        
        robot.home()
        return f"tried dropped the object to position: {dropoff_block}. Please confirm if the object is placed correctly in the image. Please proceed with the next action based on the current state."
    
    def add_task(self, task_description):
        """Add a task to the task queue, maintaining a maximum size of 5."""
        self.task_queue.append(task_description)
        if len(self.task_queue) > 5:
            self.task_queue.pop(0)  # Remove the oldest task

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
                cv2.resizeWindow("Camera Feed", 1280, 720)  # Set the desired window size
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
                    cv2.putText(frame, f"Rotation: {rotation_angle}°", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Display task queue on the frame
                    for i, task in enumerate(self.task_queue):
                        y = 120 + i * 20  # Adjust vertical position for each task
                        cv2.putText(frame, f"Task {i+1}: {task}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
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
                    # process_tool_calls
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
                    # await asyncio.sleep(1.0)
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
            print("Sending message to gemini")
            await self.session.send(input=msg)

    async def listen_audio(self):
        logger.info("Starting audio listener")
        try:
            # List available audio input devices
            num_devices = pya.get_device_count()
            logger.info("Available audio input devices:")
            for i in range(num_devices):
                device_info = pya.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                logger.info(f"Index: {i}, Name: {device_info['name']}")

            # Prompt the user to select an audio input device
            device_index = int(input("Enter the index of the desired audio input device: "))
            
            mic_info = pya.get_device_info_by_index(device_index)
            logger.debug("Using microphone: %s", mic_info["name"])
            
            self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
            )
            logger.info("Audio stream initialized successfully")
            
            while True:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            
        except Exception as e:
            logger.error("Error in audio listener: %s", str(e))
            logger.debug("Error details:", exc_info=True)

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
        if not response.text :
            logger.debug("Empty response received")
            return None

        try:
            # Clean up response text
            if (response.text.startswith('```') and response.text.endswith('```')) or response.text.startswith('json'):
                response_text = response.text.strip().strip('`')
                
                if response_text.lower().startswith('json'):
                    response_text = response_text[4:].strip()

                try:
                    tool_calls = json.loads(response_text)
                    
                    if isinstance(tool_calls, list):
                        results = []
                        for tool_call in tool_calls:
                            function_name = tool_call.get("function")
                            function_args = tool_call.get("arguments", {})
                            
                            logger.info("Processing tool call: %s", function_name)
                            logger.debug("Function arguments: %s", function_args)
                            
                            result = None  # Initialize result
                            
                            # Execute function
                            if function_name == "pickup_from_to":
                                result = await self.pickup_from_to(
                                    function_args["pickup_block"], 
                                    function_args["dropoff_block"]
                                )
                            elif function_name == "drop_off":
                                result = await self.drop_off(
                                    function_args["dropoff_block"]
                                )
                            elif function_name == "pickup_hold":
                                result = await self.pickup_hold(
                                    function_args["pickup_block"]
                                )
                            elif function_name == "home":
                                result = await self.home()
                            else:
                                logger.warning("Unknown function called: %s", function_name)
                                result = f"Unknown function: {function_name}"
                            
                            if result:
                                results.append(result)
                        
                        return "\n".join(results)  # Combine results into a single string
                    else:
                        # Handle single tool call (non-list)
                        function_name = tool_calls.get("function")
                        function_args = tool_calls.get("arguments", {})
                        
                        logger.info("Processing tool call: %s", function_name)
                        logger.debug("Function arguments: %s", function_args)
                        
                        result = None
                        
                        # Execute function
                        if function_name == "pickup_from_to":
                            result = await self.pickup_from_to(
                                function_args["pickup_block"], 
                                function_args["dropoff_block"]
                            )
                        elif function_name == "drop_off":
                            result = await self.drop_off(
                                function_args["dropoff_block"]
                            )
                        elif function_name == "pickup_hold":
                            result = await self.pickup_hold(
                                function_args["pickup_block"]
                            )
                        elif function_name == "home":
                            result = await self.home()
                        else:
                            logger.warning("Unknown function called: %s", function_name)
                            result = f"Unknown function: {function_name}"
                        
                        return result
                    
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse JSON response: %s", str(e))
                    logger.debug("Invalid JSON text: %s", response.text)
            
            elif "close" in response.text or "exit" in response.text or "stop" in response.text: 
                await self.home()
                
            else:
                # if  "Eve" in response.text or not "Eeve" in response.text or not "Eevee" in response.text:    
                await self.speak_text(response.text)
                # else:
                print(response.text)
            
                
                
        except Exception as e:
            logger.error("Error processing tool call: %s", str(e))
            logger.error(response.text)
            logger.debug("Error details:", exc_info=True)
            return None
        
    async def recieve_response(self):
        logger.info("Starting response receiver")
        
        try:
            while True:
                async for response in self.session.receive():
                    logger.debug("Response received from model")
                    
                    tool_result = await self.process_tool_calls(response)
                    
                    if tool_result:
                        logger.info("Tool execution result: %s", tool_result)
                        print("Sending message to gemini")
                        await self.session.send(
                            input=f"Function called, System generated function result: {tool_result}\n"
                                    "IF THE USER DOES NOT ASK ANYTHING TO DO. DO NOT DO ANYTHING UNTIL TOLD.",
                            end_of_turn=True
                        )
                        
                    
                        
        except Exception as e:
            logger.error("Error in response receiver: %s", str(e))
            logger.debug("Error details:", exc_info=True)

    async def run(self):
        try:
            logger.info("Starting robot control system")
            
            async with (
                client.aio.live.connect(
                    model=MODEL,
                    config=types.LiveConnectConfig(
                        response_modalities=["TEXT"],
                        system_instruction=types.Content(parts=[{"text": SYSTEM_INSTRUCTION}]),
                        tools=[self.pickup_from_to, self.drop_off, self.home, self.pickup_hold],
                    )
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                logger.info("Connected to AI model successfully")
                self.session = session
                self.is_tool_executing = False

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=2)

                logger.debug("Initializing background tasks")
                send_realtime_task = tg.create_task(self.send_realtime())
                listen_audio_task = tg.create_task(self.listen_audio())
                tg.create_task(self.get_frames())
                tg.create_task(self.recieve_response())

                await send_realtime_task
                
        except asyncio.CancelledError:
            logger.info("Shutting down robot control system")
        except Exception as e:
            logger.error("Critical error in main loop: %s", str(e))
            logger.debug("Error details:", exc_info=True)
            self.audio_stream.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera"],
    )
    args = parser.parse_args()
    
    logger.info("Starting application with mode: %s", args.mode)
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
    
    