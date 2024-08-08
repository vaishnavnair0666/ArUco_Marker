import cv2 as cv
import numpy as np

def main():
    # Load the dictionary for ArUco markers
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    parameters = cv.aruco.DetectorParameters_create()

    # Open video capture (0 for webcam, or provide a video file path)
    cap = cv.VideoCapture(1)  # Change to a video file path if needed

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect ArUco markers
        markerCorners, markerIds, _ = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        # Draw detected markers
        if markerIds is not None:
            frame = cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

            # Print detected marker IDs
            for markerId in markerIds.flatten():
                print(f"Detected marker ID: {markerId}")

        # Display the resulting frame
        cv.imshow('ArUco Marker Detection', frame)

        # Break the loop on 'o' key press
        if cv.waitKey(1) & 0xFF == ord('o'):
            break

    # Release resources
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
