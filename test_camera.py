import cv2
import time

def test_camera_and_save_video():
    # Open the default camera (usually the built-in webcam)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get the camera's default resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

    # Record for 10 seconds
    start_time = time.time()
    frame_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Write the frame
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('Camera Test', frame)

        frame_count += 1

        # Press 'q' to quit early
        if cv2.waitKey(1) == ord('q') or time.time() - start_time > 10:
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Recorded {frame_count} frames in 10 seconds.")
    print("Video saved as 'output.mp4'")

if __name__ == "__main__":
    test_camera_and_save_video()