import cv2 as cv
import numpy as np

winName = "Intruter Alert"
cv.namedWindow(winName)


# Path of the video file.
path = "/Users/gauravsingh/Desktop/AI ENGINEER/Intruder Detection/intruder_1.mp4"

# Create video capture object.
cap = cv.VideoCapture(path)

# check if the video object is opened or not.
if not cap.isOpened():
    print("Error while accessing the video file...")


# Dimensions of the video frame.
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

size = (width, height)

# Create a video writer object to save the output locally.
cap_wrt = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'MJPG'), fps, size)

# Create background subtractor object.
bg_sub = cv.createBackgroundSubtractorKNN(history = 500)

# FrameCount
frameCount = 0
intruder_status = False


# Main loop.
while True:

    frameCount += 1
    
    # Read frame one by one.
    has_frame, frame = cap.read()

    # check if there is frame to display.
    if not has_frame:
        print("No frame to display..")
        break

    # apply background subtraction to frame.
    frame_sub = bg_sub.apply(frame)

    # Apply erosion to eliminate noise.
    frame_erode = cv.erode(frame_sub, (21, 21))

    # find all the contours in the eroded frame.
    contours, hierarchy = cv.findContours(frame_erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Sort contours in desending order.
    contours_sorted = sorted(contours, key = cv.contourArea, reverse = True)

    # Check if the area of biggest contours is greater then the threshold.
    if cv.contourArea(contours_sorted[0]) > 1000:

        # draw rectangle on the biggest contour.
        box = cv.minAreaRect(contours_sorted[0])
        boxPts = np.int0(cv.boxPoints(box))
        # Draw contours.
        cv.drawContours(frame, [boxPts], -1, (0,255,0), 4)

        # get x, y, h, w from boxpts
        x, y, w, h = cv.boundingRect(boxPts)

        if x >= 1 and y >= 1:
            cv.putText(frame, "Intruder Alert!!", (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)
        
    # Display the frame.
    cv.imshow(winName, frame)
    key = cv.waitKey(1)

    # Save the output video.
    cap_wrt.write(frame)

    # Break the loop if user press 'q', 'Q' or esc key.
    if key == ord('q') or key == ord('Q') or key == 27:
        print("video ended by user..")
        break

cap.release()
cv.destroyAllWindows()
