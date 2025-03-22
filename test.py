import cv2
import sys

def test_camera(camera_path=None):
    # If no camera specified, use default camera index 0
    if camera_path is None:
        camera_path = 0
    
    # Initialize the camera with the provided path
    cap = cv2.VideoCapture(camera_path)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera at {camera_path}")
        return

    print(f"Camera at {camera_path} opened successfully. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Display the resulting frame
        cv2.imshow('Camera Test', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can specify which camera to use here
    # For the USB 2.0 PC Camera:
    # test_camera("/dev/video4")
    
    # For the USB2.0 FHD UVC WebCam:
    test_camera("/dev/video4")
    
    # Or pass no arguments to use the default camera (usually /dev/video0)
    # test_camera()