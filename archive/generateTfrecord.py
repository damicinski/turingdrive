import tensorflow as tf
import numpy as np
import scipy 
import scipy.misc
import os 
# =============================================================================
#                           JPG, CSV to tfrecord 
# =============================================================================


def generateTfrecord(imgHeight, imgWidth, imgCh, pathToImages, pathToCsv):

    imgStorage = getImageArray(pathToImages)
    labels = getCsvArray(pathToCsv)

    imgIdx = np.r_[0:len(labels)]
    np.random.shuffle(imgIdx) # Shuffle images

    writer = tf.python_io.TFRecordWriter("mnist.tfrecords")
    # iterate over each example
    numLeft = len(labels)
    for exIdx in imgIdx:
        print(numLeft)
        numLeft -= 1
        features = imgStorage[exIdx, :]
        label = labels[exIdx]

        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
              # Features contains a map of string to Feature proto objects
              feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'label': tf.train.Feature(
                    float_list=tf.train.FloatList(value=[label])),
                'image': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=features.astype("int64"))),
        }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)


    return 

def getImageArray(pathToImages):
    nrImages = len(os.listdir(pathToImages))

    imgStorage = np.zeros([nrImages, imgHeight*imgWidth*imgCh])

    idx = 0
    for file in os.listdir(pathToImages):
        if file.endswith(".jpg"):
            print(idx)
            im = scipy.misc.imread(pathToImages + file, mode = 'RGB')
            imFlat = im.reshape(imgHeight*imgWidth*imgCh)
            imgStorage[idx] = imFlat
            idx += 1

    return imgStorage

def getCsvArray(pathToCsv):
    for file in os.listdir(pathToCsv):  
        if file.endswith(".csv"):
            labels = np.genfromtxt(pathToCsv + file, delimiter=',')

    return labels[:,1]


imgHeight = 480
imgWidth = 640 
imgCh = 3

pathToImages = '/home/deepjedi/dat/udacity/challenge2/dataset2-1-160929/center_camera/'
pathToCsv = '/home/deepjedi/dat/udacity/challenge2/dataset2-1-160929/'

#pathToImages = '/Users/bergepraktik/udacityChallenge/tempData/images/'
#pathToCsv = '/Users/bergepraktik/udacityChallenge/tempData/'


generateTfrecord(imgHeight, imgWidth, imgCh, pathToImages, pathToCsv)





