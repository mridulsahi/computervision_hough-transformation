import cv2
import numpy as np
import os


from myEdgeFilterec import myEdgeFilter
from myHoughLinesec import myHoughLines
from myHoughTransformec import myHoughTransform



def findClosestIndex(array, key):
    array = np.asarray(array)
    idx = (np.abs(array - key)).argmin()
    return idx


def myHoughTransform(Im, rhoRes, thetaRes):
    ImHeight = int(np.shape(Im)[0]) 
    ImWidth = int(np.shape(Im)[1]) 
    M = np.ceil(np.sqrt(ImHeight**2 + ImWidth**2) + 1)
    rhoScale = np.arange(0,M,rhoRes)
    thetaScale = np.arange(0,np.pi*2,thetaRes)

    # we save votes here
    img_hough = np.zeros((np.shape(rhoScale)[0], np.shape(thetaScale)[0]), dtype=np.int32)
    for row in range(0,ImHeight):
        for col in range(0, ImWidth):
            if Im[row,col] > 0:
                for theta in thetaScale:
                    p = col * np.cos(theta) + row * np.sin(theta) 
                    if p >= 0:
                        rho_i = findClosestIndex(rhoScale, p)
                        theta_j = findClosestIndex(thetaScale, theta)
                        img_hough[rho_i, theta_j] +=1
    return [img_hough, rhoScale, thetaScale]

datadir    = './mydata_4.1x'      # the directory containing the images
resultsdir = './myresult_4.1x'   # the directory for dumping results

# parameters
sigma     = 2
threshold = 0.03
rhoRes    = 2
thetaRes  = np.pi / 90
nLines    = 15
# end of parameters

for file in os.listdir(datadir):
    if file.endswith('.jpg'):

        file = os.path.splitext(file)[0]
        
        # read in images
        img = cv2.imread('%s/%s.jpg' % (datadir, file))
        
        if (img.ndim == 3):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        img = np.float32(img) / 255
        
        img_edge = cv2.Canny(img, threshold1=30, threshold2=100)
        # actual Hough line code function calls
        #img_edge = myEdgeFilter(img, sigma)
        img_threshold = np.float32(img_edge > threshold)
        [img_hough, rhoScale, thetaScale] = myHoughTransform(img_threshold,rhoRes, thetaRes)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=250)

        # everything below here just saves the outputs to files
        fname = '%s/%s_01edge.png' % (resultsdir, file)
        cv2.imwrite(fname, 255 * np.sqrt(img_edge / img_edge.max()))
        
        fname = '%s/%s_02threshold.png' % (resultsdir, file)
        cv2.imwrite(fname, 255 * img_threshold)
        
        fname = '%s/%s_03hough.png' % (resultsdir, file)
        cv2.imwrite(fname, 255 * img_hough / img_hough.max())
        
        fname = '%s/%s_04lines.png' % (resultsdir, file)
        img_lines = np.dstack([img,img,img])
        # Draw lines on the image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 3)

        cv2.imwrite(fname, 255 * img_lines)
