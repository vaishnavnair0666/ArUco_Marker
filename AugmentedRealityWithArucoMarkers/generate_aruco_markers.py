import cv2 as cv
import numpy as np

# Ensure the 'aruco' module is available in your OpenCV installation
if not hasattr(cv, 'aruco'):
    raise ImportError("OpenCV is not built with the 'aruco' module.")

# Load the predefined dictionary
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Generate markers with IDs from 1 to 20
marker_size = 200

for marker_id in range(1, 21):
    # Generate the marker
    markerImage = cv.aruco.drawMarker(dictionary, marker_id, marker_size)
    
    # Define output path
    output_path = f"marker{marker_id}.png"
    
    # Save the marker image
    cv.imwrite(output_path, markerImage)
    print(f"Marker {marker_id} saved to {output_path}")
