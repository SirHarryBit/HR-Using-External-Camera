#Aim 1 - Capture video from webcam
#Aim 2 - Detect face in the video
#Aim 3 - Build gaussian pyramid
#Aim 4 - Apply fft to all frames
#aim 5 - 
#Aim  - Find Heart rate

import numpy as np
import sys
import os
import cv2

#INITIALISED PARAMETERS

# Webcam Parameters
webcam = None
realWidth = 640
realHeight = 480
videoWidth = 320
videoHeight = 240
videoChannels = 3
videoFrameRate = 15

#Video Parameters
if len(sys.argv) != 2:
    inputVideoFilename = "original.mov"
    inputVideo = cv2.VideoWriter()
    inputVideo.open(inputVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)

outputVideoFilename = "output.mov"
outputVideoWriter = cv2.VideoWriter()
outputVideoWriter.open(outputVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth//2 + 5, 30)
fontScale = 1
fontColor = (255,255,255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_default.xml"

#HELPER METHODS
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

#Program begins

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier(cascPathface)

if len(sys.argv) == 2:
    webcam = cv2.VideoCapture(sys.argv[1])
else:
    # Start the webcam
    webcam = cv2.VideoCapture(0)
webcam.set(3, realWidth)
webcam.set(4, realHeight)


while True:
    # Read the frame
    ret, frame = webcam.read()
    if ret == False:
        break

    if len(sys.argv) != 2:
        ProcessingFrame = frame.copy()
        inputVideo.write(ProcessingFrame)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    if len(sys.argv) != 2:
        cv2.imshow("Webcam Heart Rate Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)



# Release the webcam
webcam.release()

# Close all the windows
cv2.destroyAllWindows()



