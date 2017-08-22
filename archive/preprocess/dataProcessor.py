import csv
import os


class DataProcessor():

    def __init__(self, sourceFolder):
        self._sourceFolder = sourceFolder
        self._imageFolders = ['left', 'center', 'right']
        self._imageFiletype = '.pgm'
        self._originalCSV = list()

    def isInt(self,s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def createSubCsvFiles(self):
        labelFilename = self._readLabelFile()
        self._createLabelFile(labelFilename)

    def _readLabelFile(self):
        folderContent = os.listdir(self._sourceFolder)
        #Find main csv label file
        for file_ in folderContent:
            if file_.endswith('.csv'):
                labelFilename = file_
        fullSourceName = self._sourceFolder + '/' + labelFilename
        nbrOfRows = 0
        #Read main label file and store as a list of tuples
        with open(fullSourceName, 'r') as steerAngleFile:
            for row in steerAngleFile:
                timestamp, steeringAngle = row.split(',')
                if self.isInt(timestamp):
                    self._originalCSV.append((timestamp,steeringAngle.rstrip()))
                    nbrOfRows += 1
        print('Loaded %d elements'%(nbrOfRows))
        return labelFilename

    def _createLabelFile(self,labelFilename):
        #Iterate through all sub folders and create a sub csv-file in each folder
        for ii in range(len(self._imageFolders)):
            print('Processing folder: %s'%(self._imageFolders[ii]))
            folderContent = os.listdir(self._sourceFolder + '/' + self._imageFolders[ii])
            subCsvFileContent = list()
            #Create csv file
            f = open(self._sourceFolder + '/' + self._imageFolders[ii] + '/' + labelFilename.split('.')[0] + '_' + self._imageFolders[ii] + '.' + labelFilename.split('.')[1], 'w')
            writer = csv.writer(f)
            writer.writerow( ('timestamp','angle') )
            for image in folderContent:
                #Search the main csv for the closest timestamp for each image in each folder
                if image.endswith(self._imageFiletype):
                        imageName = image.split('.')[0]
                        if imageName in self._originalCSV:
                            timestampIndex = self._originalCSV.index(imageName)
                        else:
                            timestampIndex = min(range(len(self._originalCSV)), key=lambda i: abs(int(self._originalCSV[i][0])-int(imageName)))
                        timestamp = self._originalCSV[timestampIndex][0]
                        angle = self._originalCSV[timestampIndex][1]
                        subCsvFileContent.append( (timestamp, angle) )
            #Make sure that the sub csv file is sorted based on the first element in each tuple
            subCsvFileContent.sort(key=lambda tup: tup[0])
            while len(subCsvFileContent) != 0:
                item = subCsvFileContent.pop(0)
                writer.writerow( (str(item[0]), str(item[1]) ) )
            f.close()
