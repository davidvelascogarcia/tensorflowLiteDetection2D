'''
  * ************************************************************
  *      Program: Tensorflow Lite Detection 2D Module
  *      Type: Python
  *      Author: David Velasco Garcia @davidvelascogarcia
  * ************************************************************
  *
  * | INPUT PORT                           | CONTENT                                                 |
  * |--------------------------------------|---------------------------------------------------------|
  * | /tensorflowLiteDetection2D/img:i     | Input image                                             |
  *
  *
  * | OUTPUT PORT                          | CONTENT                                                 |
  * |--------------------------------------|---------------------------------------------------------|
  * | /tensorflowLiteDetection2D/img:o     | Output image with detection                             |
  * | /tensorflowLiteDetection2D/data:o    | Output result, recognition data                         |
'''

# Libraries
import argparse
import cv2
import datetime
import glob
import importlib.util
import os
import sys
import time
from threading import Thread
import numpy as np
import yarp

# Import TensorFlow / Lite libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

print("")
print("")
print("**************************************************************************")
print("**************************************************************************")
print("                 Program: Tensorflow Lite Detector 2D                     ")
print("                     Author: David Velasco Garcia                         ")
print("                             @davidvelascogarcia                          ")
print("**************************************************************************")
print("**************************************************************************")

print("")
print("Starting system ...")
print("")

print("")
print("Loading tensorflowLiteDetection2D module ...")
print("")


print("")
print("**************************************************************************")
print("YARP configuration:")
print("**************************************************************************")
print("")
print("Initializing YARP network ...")
print("")

# Init YARP Network
yarp.Network.init()

print("")
print("[INFO] Opening image input port with name /tensorflowLiteDetection2D/img:i ...")
print("")

# Open input image port
tensorflowLiteDetection2D_portIn = yarp.BufferedPortImageRgb()
tensorflowLiteDetection2D_portNameIn = '/tensorflowLiteDetection2D/img:i'
tensorflowLiteDetection2D_portIn.open(tensorflowLiteDetection2D_portNameIn)

print("")
print("[INFO] Opening image output port with name /tensorflowLiteDetection2D/img:o ...")
print("")

# Open output image port
tensorflowLiteDetection2D_portOut = yarp.Port()
tensorflowLiteDetection2D_portNameOut = '/tensorflowLiteDetection2D/img:o'
tensorflowLiteDetection2D_portOut.open(tensorflowLiteDetection2D_portNameOut)

print("")
print("[INFO] Opening data output port with name /tensorflowLiteDetection2D/data:o ...")
print("")

# Open output data port
tensorflowLiteDetection2D_portOutDet = yarp.Port()
tensorflowLiteDetection2D_portNameOutDet = '/tensorflowLiteDetection2D/data:o'
tensorflowLiteDetection2D_portOutDet.open(tensorflowLiteDetection2D_portNameOutDet)

# Create data bootle
outputBottleTensorflowLiteDetection2D = yarp.Bottle()

# Image size
image_w = 640
image_h = 480

# Prepare input image buffer
in_buf_array = np.ones((image_h, image_w, 3), np.uint8)
in_buf_image = yarp.ImageRgb()
in_buf_image.resize(image_w, image_h)
in_buf_image.setExternal(in_buf_array.data, in_buf_array.shape[1], in_buf_array.shape[0])

# Prepare output image buffer
out_buf_image = yarp.ImageRgb()
out_buf_image.resize(image_w, image_h)
out_buf_array = np.zeros((image_h, image_w, 3), np.uint8)
out_buf_image.setExternal(out_buf_array.data, out_buf_array.shape[1], out_buf_array.shape[0])

print("")
print("[INFO] YARP network configured correctly.")
print("")

print("")
print("**************************************************************************")
print("Configuration models:")
print("**************************************************************************")
print("")
print("Loading Tensorflow Lite model ...")
print("")

# Configure parser arguments
parserConfig = argparse.ArgumentParser()

parserConfig.add_argument('--dirModel', default = './../models')
parserConfig.add_argument('--graphModel', default = 'graphModel.tflite')
parserConfig.add_argument('--labelMap', default = 'labelMap.txt')
parserConfig.add_argument('--threshold', default = 0.5)

argsParser = parserConfig.parse_args()

# Set parser data
dirName = argsParser.dirModel
graphName = argsParser.graphModel
labelName = argsParser.labelMap
minThresholdConfig = float(argsParser.threshold)

# Get directory path
print("")
print("Getting directory path at " + str(datetime.datetime.now()) + " ...")
print("")
directoryPath = os.getcwd()

# Get model graph path
print("")
print("Getting graph model path at " + str(datetime.datetime.now()) + " ...")
print("")
graphModelPath = os.path.join(directoryPath, dirName, graphName)

# Get model label path
print("")
print("Getting label model path at " + str(datetime.datetime.now()) + " ...")
print("")
labelMapPath = os.path.join(directoryPath, dirName, labelName)

# Load the label map
with open(labelMapPath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

'''
Trick:
Error using the COCO "starter model", the first label with "???" name.
Delete error in the first label.
'''
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model
print("")
print("[INFO Loading model at " + str(datetime.datetime.now()) + " ...")
print("")
interpretObject = Interpreter(model_path = graphModelPath)
interpretObject.allocate_tensors()

# Get model details
inputDetails = interpretObject.get_input_details()
outputDetails = interpretObject.get_output_details()
height = inputDetails[0]['shape'][1]
width = inputDetails[0]['shape'][2]

floating_model = (inputDetails[0]['dtype'] == np.float32)

inputMean = 127.5
inputSTD = 127.5

print("")
print("[INFO] Models loaded correctly at " + str(datetime.datetime.now()) + ".")
print("")

print("")
print("**************************************************************************")
print("Waiting for input image source:")
print("**************************************************************************")
print("")
print("Waiting input image source ...")
print("")

loopControlReceiveImageSource = 0

while int(loopControlReceiveImageSource) == 0:

    print("")
    print("**************************************************************************")
    print("Processing:")
    print("**************************************************************************")
    print("")
    print("Processing data at " + str(datetime.datetime.now()) + " ...")
    print("")

    # Receive image source
    frame = tensorflowLiteDetection2D_portIn.read()

    # Buffer processed image
    in_buf_image.copy(frame)
    assert in_buf_array.__array_interface__['data'][0] == in_buf_image.getRawImage().__int__()

    # YARP -> OpenCV
    rgbFrame = in_buf_array[:, :, ::-1]

    # Prepare image
    frameResized = cv2.resize(rgbFrame, (width, height))
    inputData = np.expand_dims(frameResized, axis = 0)

    # Normalize pixel values if using a floating model
    if floating_model:
        inputData = (np.float32(inputData) - inputMean) / inputSTD

    # Set tensor
    interpretObject.set_tensor(inputDetails[0]['index'], inputData)
    interpretObject.invoke()

    # Get results from tensor
    boxes = interpretObject.get_tensor(outputDetails[0]['index'])[0]
    classes = interpretObject.get_tensor(outputDetails[1]['index'])[0]
    scores = interpretObject.get_tensor(outputDetails[2]['index'])[0]

    # Pre-configure detection values as "None":
    detectionObjectName = "None"
    detectionScore = 0
    coordinatesXY = "None, None"

    print("detection " + str(scores[0]))

    for i in range(len(scores)):

        if ((scores[i] > minThresholdConfig) and (scores[i] <= 1.0)):

            # Put detection rectangle
            yMin = int(max(1, (boxes[i][0] * image_h)))
            xMin = int(max(1, (boxes[i][1] * image_w)))
            yMax = int(min(image_h, (boxes[i][2] * image_h)))
            xMax = int(min(image_w, (boxes[i][3] * image_w)))

            # Paint rectabgle in detected object
            cv2.rectangle(in_buf_array, (xMin, yMin), (xMax, yMax), (10, 255, 0), 2)

            # Put label in OpenCV image
            detectionObjectName = labels[int(classes[i])]
            detectedObjectLabel = '%s: %d%%' % (detectionObjectName, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(detectedObjectLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            labelYMin = max(yMin, labelSize[1] + 10)

            # Paint rectangle to print name
            cv2.rectangle(in_buf_array, (xMin, labelYMin - labelSize[1] - 10), (xMin + labelSize[0], labelYMin + baseLine - 10), (255, 255, 255), cv2.FILLED)

            # Paint name in rectangle
            cv2.putText(in_buf_array, detectedObjectLabel, (xMin, labelYMin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Get centroid coordinates
            x = (xMin + xMin + labelSize[0])/2
            y = image_h - labelYMin + baseLine - 10

            # Prepare coordinates and score
            coordinatesXY = str(x) + ", " + str(y)
            detectionScore = int(scores[i] * 100)

            # Print processed data
            print("")
            print("**************************************************************************")
            print("Resume results:")
            print("**************************************************************************")
            print("")
            print("[RESULTS] tensorflowLiteDetection2D results:")
            print("")
            print("[DETECTION] Detection: " + str(detectionObjectName) + " " + str(detectionScore) + "%")
            print("[COORDINATES] Coordinates: " + "X: " + str(x) + ", Y: " + str(y))
            print("[DATE] Detection time: " + str(datetime.datetime.now()))
            print("")

            # Sending processed detection
            outputBottleTensorflowLiteDetection2D.clear()
            outputBottleTensorflowLiteDetection2D.addString("NUMBER:")
            outputBottleTensorflowLiteDetection2D.addInt(i)
            outputBottleTensorflowLiteDetection2D.addString("DETECTION:")
            outputBottleTensorflowLiteDetection2D.addString(detectionObjectName)
            outputBottleTensorflowLiteDetection2D.addString("SCORE:")
            outputBottleTensorflowLiteDetection2D.addInt(detectionScore)
            outputBottleTensorflowLiteDetection2D.addString("COORDINATES:")
            outputBottleTensorflowLiteDetection2D.addString(coordinatesXY)
            outputBottleTensorflowLiteDetection2D.addString("DATE:")
            outputBottleTensorflowLiteDetection2D.addString(str(datetime.datetime.now()))
            tensorflowLiteDetection2D_portOutDet.write(outputBottleTensorflowLiteDetection2D)

        elif scores[0] < 0.5:
            print("")
            print("[INFO] Object not detected.")
            print("")

            # Sending processed detection
            outputBottleTensorflowLiteDetection2D.clear()
            outputBottleTensorflowLiteDetection2D.addString("NUMBER:")
            outputBottleTensorflowLiteDetection2D.addInt(i)
            outputBottleTensorflowLiteDetection2D.addString("DETECTION:")
            outputBottleTensorflowLiteDetection2D.addString(detectionObjectName)
            outputBottleTensorflowLiteDetection2D.addString("SCORE:")
            outputBottleTensorflowLiteDetection2D.addInt(detectionScore)
            outputBottleTensorflowLiteDetection2D.addString("COORDINATES:")
            outputBottleTensorflowLiteDetection2D.addString(coordinatesXY)
            outputBottleTensorflowLiteDetection2D.addString("DATE:")
            outputBottleTensorflowLiteDetection2D.addString(str(datetime.datetime.now()))
            tensorflowLiteDetection2D_portOutDet.write(outputBottleTensorflowLiteDetection2D)

    # Sending processed image
    print("")
    print("[INFO] Sending processed image at " + str(datetime.datetime.now()) + " ...")
    print("")

    out_buf_array[:,:] = in_buf_array
    tensorflowLiteDetection2D_portOut.write(out_buf_image)

# Close YARP ports
print("[INFO] Closing ports ...")
tensorflowLiteDetection2D_portIn.close()
tensorflowLiteDetection2D_portOut.close()
tensorflowLiteDetection2D_portOutDet.close()

print("")
print("")
print("**************************************************************************")
print("Program finished")
print("**************************************************************************")
print("")
print("tensorflowLiteDetection2D program finished correctly.")
print("")
