import cv2 as cv
import numpy as np

def overlayAruco(markerCorners, markerIds, img, marker_image_map, scaleFactor=0.5, alpha=0.7, drawId=True):
    imgCopy = img.copy()

    for i, corners in enumerate(markerCorners):
        markerId = markerIds[i][0]

        if markerId not in marker_image_map:
            continue

        imgOverlay = cv.imread(marker_image_map[markerId], cv.IMREAD_UNCHANGED)
        if imgOverlay is None:
            print(f"Error: Could not load the overlay image for marker ID {markerId}.")
            continue

        if imgOverlay.shape[2] == 4:  # Check if the overlay image has an alpha channel
            imgOverlay = cv.cvtColor(imgOverlay, cv.COLOR_BGRA2BGR)
        else:
            imgOverlay = cv.cvtColor(imgOverlay, cv.COLOR_BGR2BGRA)  # Add an alpha channel

        corners = corners.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        topLeft = tuple(map(int, topLeft))
        bottomRight = tuple(map(int, bottomRight))

        markerWidth = int(np.linalg.norm(bottomRight - bottomLeft))
        markerHeight = int(np.linalg.norm(topRight - bottomRight))

        overlayWidth = int(markerWidth * scaleFactor)
        overlayHeight = int(markerHeight * scaleFactor)

        overlayResized = cv.resize(imgOverlay, (overlayWidth, overlayHeight))

        x1 = int(topLeft[0] - (overlayWidth - markerWidth) / 2)
        y1 = int(topLeft[1] - (overlayHeight - markerHeight) / 2)

        x2 = x1 + overlayWidth
        y2 = y1 + overlayHeight

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        imgOverlayResized = overlayResized[:y2-y1, :x2-x1]
        alphaOverlay = imgOverlayResized[:, :, 3] / 255.0  # Extract alpha channel

        for c in range(0, 3):
            imgCopy[y1:y2, x1:x2, c] = alphaOverlay * imgOverlayResized[:, :, c] + (1 - alphaOverlay) * imgCopy[y1:y2, x1:x2, c]

        cv.rectangle(imgCopy, topLeft, bottomRight, (0, 255, 0), 2)

        if drawId:
            cv.putText(imgCopy, str(markerId), (topLeft[0], topLeft[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return imgCopy



def main():
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    parameters = cv.aruco.DetectorParameters_create()

    cap = cv.VideoCapture(1)

    marker_image_map = {
        1: 'Pizza.jpg',
        3: 'cheese.jpg',
        2: 'tomato.jpg',
        4: 'mushroom.jpg',
        5: 'burger.jpg',
        6: 'monalisa.jpg',
        7: 'Art (1).jpg',
        8: 'Art (2).jpg',
        9: 'Art (3).jpg',
        10: 'Art (4).jpg',
        11: 'Art (5).jpg',
        12: 'Art (6).jpg',
        13: 'Art (7).jpg',
        14: 'Art (8).jpg',
        15: 'Art (9).jpg',
        16: 'Art (10).jpg',
        17: 'Art (11).jpg',
        18: 'Art (12).jpg',
        19: 'Art (13).jpg',
        20: 'monalisa.jpg',
        33: 'new_scenery.jpg'
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        markerCorners, markerIds, _ = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        if markerIds is not None:
            frame = overlayAruco(markerCorners, markerIds, frame, marker_image_map, scaleFactor=1.5)

            for markerId in markerIds.flatten():
                print(f"Detected marker ID: {markerId}")

        cv.imshow('ArUco Marker Detection', frame)

        if cv.waitKey(1) & 0xFF == ord('o'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

