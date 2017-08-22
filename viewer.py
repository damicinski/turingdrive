import csv
#import math
import os
import pygame

class Viewer():
    ''' Class for visualising prediction and ground truth as overlay to the input video.
    Expects a folder with png images and two CSV files.
    The CSV files represents predictions and ground truth.
    The frame IDs should match between the files per line. The frame IDs should match the image file names.
    CSV file with recorded data should have the format:
        index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt
    CSV file with predicted data should have the format:
        index, angle '''
    def __init__(self, sourceFolder, predFile, recFile):
        self._sourceFolder = sourceFolder
        self._predFile = predFile
        self._recFile = recFile

        self._videoWidth = 640
        self._videoHeight = 480
        self._pixelsPerMeter = self._videoWidth / 2.5 # Assumption
        self._imageFileType = '.png'
        self._frameTime = 100 # ms
        self._fontSize = 24

        self._screen, self._cameraSurface, self._font = self._initPygame(self._videoWidth, self._videoHeight)

    def play(self):
        # Load CSV files
        try:
            f = open(self._sourceFolder + '/' + self._predFile, 'rb')
            pFile = csv.reader(f)
            f = open(self._sourceFolder + '/' + self._recFile, 'rb')
            rFile = csv.reader(f)

            # Skip info line
            rFile.next()
            pFile.next()
        except ValueError:
            print "Exception 1"
            print ValueError
            return False

        # Go through all frames and draw them with their overlay
        while True:
            # Read line
            success = False
            try:
                rRow = rFile.next()
                pRow = pFile.next()
                if rRow[0] == pRow[0]: # Same frame ID
                    image = pygame.image.load(self._sourceFolder + '/' + rRow[0] + self._imageFileType).convert()
                    success = True
            except:
                success = False
                print "Done"
            if success == False:
                break

            # Show image
            self._screen.blit(image, (0, 0))

            # Render curve overlay
            p = float(pRow[1])
            r = float(rRow[6])
#            speed = float(rRow[8])

#            radius = int(100 * self._pixelsPerMeter)

#            posX = self._videoWidth / 2 + radius
#            posY = self._videoHeight

#            pygame.draw.circle(self._screen, (0, 0, 255), (posX, posY), radius, 2)

            # Show info text
            e = p - r
            text = self._font.render("Predicted = %.3f, Recorded = %.3f, Error = %.3f" % (p, r, e), 1, (255, 0, 0))
            self._screen.blit(text, (0, self._videoHeight - self._fontSize))

            # Display
            pygame.display.update()

            # Wait until next frame
            pygame.time.delay(self._frameTime)

    def _initPygame(self, width, height):
        pygame.init()
        font = pygame.font.Font(None, self._fontSize)
        size = (width, height)
        pygame.display.set_caption('Berge Turing Drive')
        screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
        cameraSurface = pygame.surface.Surface(size, 0, 24).convert()
        return screen, cameraSurface, font

# End class Viewer

v = Viewer('Example', 'pred.csv', 'rec.csv')
v.play()
