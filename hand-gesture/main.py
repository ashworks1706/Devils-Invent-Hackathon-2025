import cv2
import mediapipe as mp
import numpy as np
import logging
from mediapipe.framework.formats import landmark_pb2
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # filename='hand_tracking.log',
    filemode='a'
)
logger = logging.getLogger('hand_gesture_tracker')

# Initialize MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def process_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """Process the hand tracking result."""
    global image
    
    try:
        if result.hand_world_landmarks:
            try:
                hand_world_landmarks = result.hand_world_landmarks[0]
                hand_landmarks = result.hand_landmarks[0]  # For 2D display position
                
                palm_x = hand_world_landmarks[0].x
                palm_y = hand_world_landmarks[0].y
                palm_z = hand_world_landmarks[0].z
                
                grip_state = "GRAB" if is_fist(hand_world_landmarks) else "OPEN"
                
                # Log palm position and grip state
                logger.info(f"Palm position: x={palm_x:.3f}, y={palm_y:.3f}, z={palm_z:.3f}, State: {grip_state}")
                
                try:
                    # Get 2D screen position for palm (using wrist landmark)
                    h, w, _ = image.shape
                    palm_screen_x = int(hand_landmarks[0].x * w)
                    palm_screen_y = int(hand_landmarks[0].y * h)
                    
                    # Draw text above the palm
                    position_text = f"x={palm_x:.2f}, y={palm_y:.2f}, z={palm_z:.2f}"
                    state_text = f"State: {grip_state}"
                    
                    # Position text above the palm
                    cv2.putText(image, position_text, (palm_screen_x - 70, palm_screen_y - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(image, state_text, (palm_screen_x - 50, palm_screen_y - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    logger.info(f"cv2.putText(image, {position_text}, ({palm_screen_x - 70}, {palm_screen_y - 30}), "
                                "cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)")
                    
                    # Draw a circle at the wrist position for reference
                    cv2.circle(image, (palm_screen_x, palm_screen_y), 5, (0, 255, 0), -1)
                    
                    
                except Exception as drawing_error:
                    logger.error(f"Error drawing on image: {str(drawing_error)}")
            except IndexError as idx_error:
                logger.error(f"Index error accessing landmarks: {str(idx_error)}")
            except Exception as landmark_error:
                logger.error(f"Error processing landmarks: {str(landmark_error)}")
        else:
            logger.warning("No hand landmarks detected in this frame")
    except Exception as e:
        logger.error(f"Unexpected error in process_result: {str(e)}", exc_info=True)


logger.info("Starting hand gesture tracking application")
        
# Create a hand landmarker instance with the live stream mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=process_result)

logger.info("Hand landmarker options configured")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Failed to open webcam")
else:
    logger.info("Webcam initialized successfully")

def is_fist(world_landmarks):
    """Detect if hand is making a fist/grab gesture."""
    tips = [8, 12, 16, 20]  # Fingertip indices
    mcp = [5, 9, 13, 17]    # Metacarpophalangeal joint indices
    
    is_closed = True
    for tip, base in zip(tips, mcp):
        if (world_landmarks[tip].y < world_landmarks[base].y):
            is_closed = False
            break
    
    return is_closed

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        logger.info("Hand landmarker created successfully")
        timestamp = 0  # Manual timestamp counter
        session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Tracking session started at {session_start}")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                logger.error("Failed to read frame from webcam")
                break

            image = cv2.flip(image, 1)  # Mirror display
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Send for processing with MONOTONICALLY INCREASING timestamp
            landmarker.detect_async(mp_image, timestamp)
            timestamp += 1  # Increment timestamp
            
            cv2.imshow('MediaPipe Hand Tracking', image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                logger.info("ESC key pressed, exiting")
                break
            
            
except Exception as e:
    logger.error(f"Error in hand tracking: {str(e)}", exc_info=True)
finally:
    logger.info("Cleaning up resources")
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Application terminated")
