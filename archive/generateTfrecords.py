import tensorflow as tf
import numpy as np
import scipy 
import scipy.misc
import os 
# =============================================================================
#                           JPG, CSV to tfrecords 
# =============================================================================
'''
    Converts a directiry of images and associated .csv file with labels into 
    several tfrecords, the number of which is decied by the parameter 
    nrOfSplits
'''
# =============================================================================
#                           Helper Functions 
# =============================================================================

def generateTfrecord2(imgHeight, imgWidth, imgCh, pathToImages, listOfImages, csvEntries, nr):

    imgStorage = getImageArray2(pathToImages, listOfImages)

    labels = csvEntries
    imgIdx = np.r_[0:len(labels)]

    writer = tf.python_io.TFRecordWriter("turing" + str(nr) + ".tfrecords")
    # iterate over each example
    numLeft = len(labels)

    print "Generating record: " + str(nr) + " Number of images: " + str(len(listOfImages))
    for exIdx in imgIdx:
        print "Images left:" + str(numLeft)
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
                    float_list=tf.train.FloatList(value = [label])),
                'image': tf.train.Feature(
                    int64_list=tf.train.Int64List(value = features.astype("int64"))),
        }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)

    return


def getImageArray2(pathToImages, listOfImages):
    nrImages = len(listOfImages)
    imgStorage = np.zeros([nrImages, imgHeight*imgWidth*imgCh])

    idx = 0
    for file in listOfImages:        
        im = scipy.misc.imread(pathToImages + file, mode = 'RGB')
        imFlat = im.reshape(imgHeight*imgWidth*imgCh)
        imgStorage[idx] = imFlat
        idx += 1

    print "Loaded images into memory"
    return imgStorage



def getCsvArray(pathToCsv):
    for file in os.listdir(pathToCsv):  
        if file.endswith(".csv"):
            labels = np.genfromtxt(pathToCsv + file, delimiter=',')

    return labels[:,1]

# =============================================================================
#                           Parameters
# =============================================================================

imgHeight = 480
imgWidth = 640 
imgCh = 3
nrOfSplits = 100 # Into how many tfrecords the images should be split

pathToImages = '/Users/bergepraktik/udacityChallenge/dataset2-1-160929/center_camera/'
pathToCsv = '/Users/bergepraktik/udacityChallenge/dataset2-1-160929/'

#pathToImages = '/Users/bergepraktik/udacityChallenge/tempData/images/'
#pathToCsv = '/Users/bergepraktik/udacityChallenge/tempData/'


# =============================================================================
#                           Script
# =============================================================================

# Make a list of all .jpgs 
listOfImages = os.listdir(pathToImages)

# Remove any files that aren't .jpg
for file in listOfImages:
    if not file.endswith(".jpg"):
        listOfImages.remove(file)

# Read the entire array
csvArray  = getCsvArray(pathToCsv)

if not (len(csvArray) == len(listOfImages)):
    print 'nr of csv entires does not equal the number of images'

# The index of images/csv-entries 
indexList = np.r_[0:len(csvArray)]
# Shuffle the index around 
np.random.shuffle(indexList)

# split the shuffled indices 
splitIndices = np.array_split(indexList, nrOfSplits)

print "Start generating"
idx = 1
for singleSplitIndices in splitIndices:
    subListOfImages = []
    subCsvArray = []
    for elem in singleSplitIndices:
#        print elem
        subListOfImages.append(listOfImages[elem])        
        subCsvArray.append(csvArray[elem])

#    print "Generate a record for:"
#    print subListOfImages
#    print subCsvArray
    generateTfrecord2(imgHeight, imgWidth, imgCh, pathToImages, subListOfImages, csvArray[singleSplitIndices], idx)

    idx += 1



























