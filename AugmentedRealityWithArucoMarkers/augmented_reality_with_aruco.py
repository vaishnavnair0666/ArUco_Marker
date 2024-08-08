import cv2 as cv
import numpy as np
import argparse
import sys
import os

def main(args):
    # Load the source image for augmentation
    im_src = cv.imread("new_scenery.jpg")

    # Set output file name based on input type
    outputFile = "ar_out_py2.avi"
    if args.image:
        if not os.path.isfile(args.image):
            print(f"Input image file {args.image} doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.image)
        outputFile = args.image[:-4] + '_ar_out_py.jpg'
    elif args.video:
        if not os.path.isfile(args.video):
            print(f"Input video file {args.video} doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        outputFile = args.video[:-4] + '_ar_out_py.avi'
        print(f"Storing it as: {outputFile}")
    else:
        cap = cv.VideoCapture(0)

    # Get the video writer initialized to save the output video
    if not args.image:
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28, 
                                    (round(2 * cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    winName = "Augmented Reality using Aruco markers in OpenCV"
    
    while cv.waitKey(1) < 0:
        try:
            # Get frame from the video
            hasFrame, frame = cap.read()

            # Stop the program if end of video is reached
            if not hasFrame:
                print(f"Done processing! Output file is stored as {outputFile}")
                cv.waitKey(3000)
                break

            # Load the dictionary that was used to generate the markers
            dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
            parameters = cv.aruco.DetectorParameters_create()

            # Detect the markers in the image
            markerCorners, markerIds, _ = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)

            if markerIds is not None and len(markerIds) >= 4:
                # Define points for the new scenery image to be overlaid
                pts_src = np.array([[0, 0], [im_src.shape[1], 0], [im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]], dtype=np.float32)
                pts_dst = []

                # Collect the corners of the four markers
                for marker_id in [0, 1, 2, 3]:
                    try:
                        index = np.where(markerIds == marker_id)[0][0]
                        corners = markerCorners[index][0]
                        pts_dst.append(corners[0])
                    except IndexError:
                        print(f"Marker {marker_id} not detected.")
                        break
                
                if len(pts_dst) == 4:
                    pts_dst = np.array(pts_dst, dtype=np.float32)

                    # Calculate the homography
                    h, _ = cv.findHomography(pts_src, pts_dst)

                    # Warp source image to destination based on homography
                    warped_image = cv.warpPerspective(im_src, h, (frame.shape[1], frame.shape[0]))

                    # Prepare a mask representing region to copy from the warped image into the original frame
                    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    cv.fillConvexPoly(mask, np.int32([pts_dst]), (255, 255, 255), cv.LINE_AA)

                    # Erode the mask to not copy the boundary effects from the warping
                    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
                    mask = cv.erode(mask, element, iterations=3)

                    # Copy the mask into 3 channels
                    mask3 = np.zeros_like(warped_image)
                    for i in range(3):
                        mask3[:, :, i] = mask / 255

                    # Copy the warped image into the original frame in the mask region
                    warped_image = warped_image.astype(float)
                    frame = frame.astype(float)
                    warped_image_masked = cv.multiply(warped_image, mask3)
                    frame_masked = cv.multiply(frame, 1 - mask3)
                    im_out = cv.add(warped_image_masked, frame_masked)

                    # Show the original image and the new output image side by side
                    concatenatedOutput = cv.hconcat([frame, im_out]).astype(np.uint8)
                    cv.imshow(winName, concatenatedOutput)

                    # Write the frame with the detection boxes
                    if args.image:
                        cv.imwrite(outputFile, concatenatedOutput)
                    else:
                        vid_writer.write(concatenatedOutput)
                else:
                    cv.imshow(winName, frame)
            else:
                cv.imshow(winName, frame)

        except Exception as e:
            print(e)
            cv.imshow(winName, frame)

    cap.release()
    if 'vid_writer' in locals():
        vid_writer.release()
        print('Video writer released.')

    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augmented Reality using Aruco markers in OpenCV')
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--video', help='Path to video file.')
    args = parser.parse_args()
    winName = "Augmented Reality using Aruco markers in OpenCV"
    main(args)
