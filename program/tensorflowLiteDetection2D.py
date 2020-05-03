'''
 * ************************************************************
 *      Program: Tensorflow Lite Detection 2D Module
 *      Type: Python
 *      Author: David Velasco Garcia @davidvelascogarcia
 * ************************************************************
 */

/*
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
  * | /tensorflowLiteDetection2D/coord:o   | Output result, recognition coordinates                  |

  *
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



print("**************************************************************************")
print("**************************************************************************")
print("                 Program: Tensorflow Lite Detector 2D                     ")
print("                     Author: David Velasco Garcia                         ")
print("                             @davidvelascogarcia                          ")
print("**************************************************************************")
print("**************************************************************************")

print("")
print("Starting system...")

print("")
print("Loading tensorflowLiteDetection2D module...")



print("")
print("")
print("**************************************************************************")
print("YARP configuration:")
print("**************************************************************************")
print("")
print("Initializing YARP network...")

# Init YARP Network
yarp.Network.init()


print("")
print("Opening image input port with name /tensorflowLiteDetection2D/img:i...")

# Open input image port
tensorflowLiteDetection2D_portIn = yarp.BufferedPortImageRgb()
tensorflowLiteDetection2D_portNameIn = '/tensorflowLiteDetection2D/img:i'
tensorflowLiteDetection2D_portIn.open(tensorflowLiteDetection2D_portNameIn)

print("")
print("Opening image output port with name /tensorflowLiteDetection2D/img:o...")

# Open output image port
tensorflowLiteDetection2D_portOut = yarp.Port()
tensorflowLiteDetection2D_portNameOut = '/tensorflowLiteDetection2D/img:o'
tensorflowLiteDetection2D_portOut.open(tensorflowLiteDetection2D_portNameOut)

print("")
print("Opening data output port with name /tensorflowLiteDetection2D/data:o...")

# Open output data port
tensorflowLiteDetection2D_portOutDet = yarp.Port()
tensorflowLiteDetection2D_portNameOutDet = '/tensorflowLiteDetection2D/data:o'
tensorflowLiteDetection2D_portOutDet.open(tensorflowLiteDetection2D_portNameOutDet)

print("")
print("Opening data output port with name /tensorflowLiteDetection2D/coord:o...")

# Open output coordinates data port
tensorflowLiteDetection2D_portOutCoord = yarp.Port()
tensorflowLiteDetection2D_portNameOutCoord = '/tensorflowLiteDetection2D/coord:o'
tensorflowLiteDetection2D_portOutCoord.open(tensorflowLiteDetection2D_portNameOutCoord)

# Create data bootle
cmd=yarp.Bottle()

# Create coordinates bootle
coordinates=yarp.Bottle()

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
print("")
print("**************************************************************************")
print("Configuration models:")
print("**************************************************************************")
print("")
print("Loading Tensorflow Lite model...")

# Configure parser arguments
parserConfig = argparse.ArgumentParser()

parserConfig.add_argument('--dirModel', default='./../models')
parserConfig.add_argument('--graphModel', default='graphModel.tflite')
parserConfig.add_argument('--labelMap', default='labelMap.txt')
parserConfig.add_argument('--threshold', default=0.5)

argsParser = parserConfig.parse_args()

# Set parser data
dirName = argsParser.dirModel
graphName = argsParser.graphModel
labelName = argsParser.labelMap
minThresholdConfig = float(argsParser.threshold)

# Get directory path
print("")
print("Getting directory path...")
directoryPath = os.getcwd()

# Get model graph path
print("")
print("Getting graph model path...")
graphModelPath = os.path.join(directoryPath,dirName,graphName)

# Get model label path
print("")
print("Getting label model path...")
labelMapPath = os.path.join(directoryPath,dirName,labelName)

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
print("Loading model...")
interpretObject = Interpreter(model_path=graphModelPath)

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
print("Waiting input image source...")
print("")
print("")
print("**************************************************************************")
print("Processing:")
print("**************************************************************************")
while True:

    # Recieve image source
    frame = tensorflowLiteDetection2D_portIn.read()

    # Buffer processed image
    in_buf_image.copy(frame)
    assert in_buf_array.__array_interface__['data'][0] == in_buf_image.getRawImage().__int__()

    # YARP -> OpenCV
    rgb_frame = in_buf_array[:, :, ::-1]

    # Prepare image
    frame_rgb = rgb_frame
    frame_resized = cv2.resize(frame_rgb, (width, height))
    inputData = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        inputData = (np.float32(inputData) - inputMean) / inputSTD

    # Set tensor
    interpretObject.set_tensor(inputDetails[0]['index'],inputData)
    interpretObject.invoke()

    # Get results from tensor
    boxes = interpretObject.get_tensor(outputDetails[0]['index'])[0]
    classes = interpretObject.get_tensor(outputDetails[1]['index'])[0]
    scores = interpretObject.get_tensor(outputDetails[2]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > minThresholdConfig) and (scores[i] <= 1.0)):

            # Put detection rectangle
            ymin = int(max(1,(boxes[i][0] * image_h)))
            xmin = int(max(1,(boxes[i][1] * image_w)))
            ymax = int(min(image_h,(boxes[i][2] * image_h)))
            xmax = int(min(image_w,(boxes[i][3] * image_w)))

            cv2.rectangle(in_buf_array, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Put label in OpenCV image
            detectionObjectName = labels[int(classes[i])]
            label = '%s: %d%%' % (detectionObjectName, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(in_buf_array, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(in_buf_array, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


            x=(xmin+xmin+labelSize[0])/2
            y=image_h-label_ymin+baseLine-10

            # Get detection time
            detectionTime = datetime.datetime.now()

            # Prepare coordinates and score
            coordinatesXY=str(x)+", "+str(y)
            detectionScore=int(scores[i]*100)

            # Print processed data
            print("")
            print("")
            print("**************************************************************************")
            print("Resume:")
            print("**************************************************************************")
            print ("")
            print ("Detection: "+str(detectionObjectName)+" "+str(detectionScore)+"%")
            print("Coordinates:")
            print("X: ", x)
            print("Y: ", y)
            print("Detection time: "+str(detectionTime))

            # Sending processed detection
            cmd.clear()
            cmd.addString("Detection number:")
            cmd.addInt(i)
            cmd.addString("Detection:")
            cmd.addString(detectionObjectName)
            cmd.addString("Score:")
            cmd.addInt(detectionScore)
            cmd.addString("Coordinates:")
            cmd.addString(coordinatesXY)
            cmd.addString("Detection time:")
            cmd.addString(str(detectionTime))
            tensorflowLiteDetection2D_portOutDet.write(cmd)

            # Sending coordinates detection
            coordinates.clear()
            coordinates.addString("X: ")
            coordinates.addString(str(x))
            coordinates.addString("Y: ")
            coordinates.addString(str(y))
            tensorflowLiteDetection2D_portOutCoord.write(coordinates)

    # Sending processed image
    print("")
    print ('Sending processed image...')
    out_buf_array[:,:] = in_buf_array
    tensorflowLiteDetection2D_portOut.write(out_buf_image)


# Close YARP ports
print ('Closing ports...')
tensorflowLiteDetection2D_portIn.close()
tensorflowLiteDetection2D_portOut.close()
tensorflowLiteDetection2D_portOutDet.close()
tensorflowLiteDetection2D_portOutCoord.close()

print("")
print("")
print("**************************************************************************")
print("Program finished")
print("**************************************************************************")
