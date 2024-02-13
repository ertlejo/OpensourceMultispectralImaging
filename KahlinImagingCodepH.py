import arducam_mipicamera as arducam #importing libraries
import RPi.GPIO as GPIO 
import v4l2 #sudo pip install v4l2
import time
import os
import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

''' 
'led'  = 'GREEN'  = 'GR' (GPIO pin 32)
'led2' = 'RED'    = 'RD'  (GPIO pin 29)
'led3' = 'IR'     = 'IFR' (GPIO pin 35)
'led4' = 'BLUE'   = 'BL'  (GPIO pin 31)
       = 'Chl_fl'  

mask is inverted because our system has white background
'ivt'     = inverted mask image
'masking' = binary image from 'ivt'     
     
'''

GPIO.setmode(GPIO.BOARD) #listen for pins
GPIO.setwarnings(False)
GPIO.setup(32, GPIO.OUT) #GREEN; call pin 32 for use
GPIO.setup(29, GPIO.OUT) #RED; call pin 33 for use
GPIO.setup(35, GPIO.OUT) #IR; call pin 35 to use
GPIO.setup(31, GPIO.OUT) #BLUE; call pin 31 to use
GPIO.setup(11, GPIO.OUT) #Filter  servo, use pin 11
GPIO.setup(13, GPIO.OUT) #Servo Relay, Pin 13
led = 32 #GREEN; assign pin 32 to variable 'led'
led2 = 29 #RED; assign pin 33 to variable 'led2'
led3 = 35 #IR; assign pin 35 to variable 'led3'
led4 = 31 #BLUE; assign pin 31 to variable 'led4'
p=GPIO.PWM(11, 50) #GPIO 3 for PWM at 50hz
p.start(0)#initialization for servo

#Make sure all LEDs are off by setting all relevant pins high
GPIO.output(led, GPIO.HIGH)#GREEN
GPIO.output(led2, GPIO.HIGH)#RED
GPIO.output(led3, GPIO.HIGH)#IR
GPIO.output(led4, GPIO.HIGH)#BLUE

#plant number, 1-8
Rep = "1" #1-8
TRT = "4" #4, 5.5, 7, 8.5
Cult = "Supercascade_Red" #Pink Stakes
#Cult = "Carpet_Blue" #Yellow Stakes
#Cult = "Wave_Purple" #Orange Stakes

Date = "10_4" #9_20, 9_22, 9_25, 9_27, 9_29, 10_2, 10_4, 10_6, 10_9 

#Normal
timestamp = ","+Cult+","+TRT+","+Rep+","
parent_dir = "/home/pi/Desktop/MultiSpectralImaging/Kahlin/pHv3/"#"+Date+"/"+Cult+"/"+TRT+"/"+Rep+"/"# Parent Directory, has the stake color and fert rate, 0, 0.25, 0.5, 1, 2
directory = str(os.path.join(parent_dir, Date, Cult, TRT, Rep))# directory for the IEMI (index analyzer): no '/' in the directory
drct =  str(os.path.join(parent_dir, Date, Cult, TRT, Rep)+"/")#directory for taking images: have '/' in drct

#size Calibration
#timestamp = "SIZE"+Date
#parent_dir = "/home/pi/Desktop/MultiSpectralImaging/Kahlin/pHv3/SizeCal/"#+Date+"/"+Cult+"/"+TRT+"/"+Rep+"/"# Parent Directory, has the stake color and fert rate, 0, 0.25, 0.5, 1, 2
#directory = str(os.path.join(parent_dir, timestamp))# directory for the IEMI (index analyzer): no '/' in the directory
#drct =  str(os.path.join(parent_dir, timestamp)+"/")#directory for taking images: have '/' in drct


os.mkdir(directory)# Create the directory

camera = arducam.mipi_camera() #name the camera
print("Open camera...")
camera.init_camera() #turn on the camera
print("Setting the resolution...")
fmt = camera.set_resolution(1280, 800) #set resolution
print("Current resolution is {}".format(fmt))
time.sleep(0.1) #wait for 0.1 second
#camera.set_control(v4l2.V4L2_CID_GAIN, 1) #range 0-15
    
def ImageGenerator(ExposureTime, Gain, LED_col, Image_col):
    try:
        #print("Setting the Exposure...")
        camera.set_control(v4l2.V4L2_CID_EXPOSURE, ExposureTime) #Set relevant exposure time
        #print("Setting the Gain...")
        camera.set_control(v4l2.V4L2_CID_GAIN, Gain)
        camera.software_auto_exposure(enable = False)
    except Exception as e:
        print(e)
    GPIO.output(LED_col, GPIO.LOW) #set led for first picture
    time.sleep(0.2) #wait 0.2 seconds
    fmt = (timestamp, Image_col) #set image title     
    frame = camera.capture(encoding = 'jpeg') #take image and store as 'jepg'
    frame.as_array.tofile(drct+"{}x{}.jpg".format(fmt[0],fmt[1])) #finalize naming scheme and apply name for storage
    GPIO.output(LED_col, GPIO.HIGH) #turn off led

#Generate monochrome images under different light conditions
p.ChangeDutyCycle(2.2) #set servo to zero point
time.sleep(1)#wait for servo to be at 01128
ImageGenerator(1842, 1, led2, "RED")#make a red image
ImageGenerator(2494, 1, led3, "IR")#make a infrared image2356
ImageGenerator(1176, 1, led4, "BLUE")#make a blue image
ImageGenerator(2503, 1, led, "GREEN")#make a green image
p.ChangeDutyCycle(4)
time.sleep(1)#wait for servo motion
ImageGenerator(55000, 1, led4, "Chl_fl")#make a mask image
p.ChangeDutyCycle(2.2) #set servo to zero point
time.sleep(1) #wait for servo to return to zero point
p.ChangeDutyCycle(0)

camera.close_camera() #turn off camera

def IEMI(minPxs, maxPxs, MinTHR, csvname, imageformat, folder):
    filePath = folder# tells program where to look for images. D0 NOT change. If you need to change the folder, do so near the bottom of the program.
    # Make sure to specify the correct file extension for your images!!!
    fileList=glob.glob(filePath+imageformat)
    
    for a in fileList[::-1]:# A for loop if statement to extract all file list except 'histogram.png' and 'filterd.png'
        if a.find('_filtered.png')>-1 or a.find('_histogram.png')>-1: #if the list contains a filename matches with these texts
            fileList.remove(a)#remove these filenames from the list

    with open(csvname, 'a', newline='') as csvfile: #to create csv file which will include the data
        writer = csv.DictWriter(csvfile, fieldnames = ["File name","Avg_Hist_ACI", "Std_Hist_ACI", "Avg_Hist_NDVI", "Std_Hist_NDVI", "Area_cm2", "Area_Pixels", "Threshold"]) #header of each column within the csv file
        writer.writeheader()
        
    for fdx, filename in enumerate(fileList): #A For loop statment: iteration from all indices (the filenames) within the folder
        fname = filename.rsplit(".", 1)[0] #Treating the string into a list following the separator "."    
        #print(fname)
        if filename.find('Chl_fl')!= -1:#mask = Chlorophyll fluorescence image with a masked background
        #if filename.find('IR')!= -1:#mask = Infra-red image with a masked background
            #img = cv2.imread(filename) #Define 'img' that reads an image corresponding to a filename;
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Although we use generally grayscale image, for color or RGB image, convert image as grayscale.
            #ivt = cv2.bitwise_not(img)#inverted mask image
            #cv2.imwrite(str(fname)+"_inverted.png",ivt)
            Hist = cv2.calcHist([img], [0], None, [256], [0,255]) #make a histogram of pixel intensity
            #Hist = Hist*8.53
            plt.plot(Hist),plt.yscale('log'),plt.xlabel('Intensity'),plt.ylabel('Pixel frequency'),plt.title('Histogram of inverted mask image'),plt.savefig(str(fname)+"_histogram.png"),plt.close()

            # Find the location in the histogram with the lowest pixel intensity (within a specified range of pixel intensities)
            # Underlying assumption is that the lowest value between background and plant is somewhere 
            # at a pixel intensity between 20 and 90. That range can be easily adjusted in the following instruction.
            # If you change the lower pixel intensity at which to start looking for a minimum, make sure to change that value
            # both after the '=' sign and in the Hist[xx:90] range.
            # print(Hist) This instruction can be used to output the histogram
            #Thr = MinTHR + np.argmin(Hist[MinTHR:10])
            Thr = MinTHR + np.argmin(Hist[MinTHR:12])
            print(Thr)
           # Thr=3
            #threshold the grayscaled-image as two channel (black and white) based on threshold values; 'ret': return
            ret, thres = cv2.threshold(img, Thr, 255, cv2.THRESH_BINARY)
        
            #The function yields the information of the every seperated components from the thresholded two-channel-image
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thres)
            #labels: matrix size, stats: the stats in the matrix, centroids: x and y locations within the matrix
        
            #CC_STAT_AREA: function to get area from the stats in the components of the image. it can be changed to width or height of the image; please refer https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
            areas = stats[1:,cv2.CC_STAT_AREA]#with the 'stats' from the two-channel treshold, area can be calculated.
            result = np.zeros((labels.shape), np.uint8)#empty matrix, will be used to write the thresholded two-channel image
 
            #For loop statement to remove pixels outside of the given range; associated to first two parameter of line 114 
            for i in range(0, nlabels - 1):    
                if areas[i] > minPxs and areas[i] <=maxPxs: #if the components within the image meets the conditions, keep and others are discarded
                    result[labels == i + 1] = 255 #convert the only components meeting the conditions
            cv2.imwrite(str(fname)+"_filtered.png",result)#save the background image
            masking = cv2.imread(fname+"_filtered.png", cv2.IMREAD_UNCHANGED)#open background image without format change
            PCS = cv2.countNonZero(masking)#projected canopy size in # of pixel
            cm_area = (PCS*112)/99509#One Off Calibration for Area
        elif filename.find("RED") != -1:#Only find an image including the "Red"
            Pch0 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) #Read the monochrome image at red light and define it as Pch0;
            RD = np.float64(Pch0) #generate RD (Red) objects as arrays in float64 format from monochrom image under red light; 
        elif filename.find('IR')!= -1:#IR = "Infrared"
            Pch1 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            IFR = np.float64(Pch1)
        elif filename.find('BLUE')!= -1:#BLUE
            Pch2 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            BL = np.float64(Pch2) 
        elif filename.find('GREEN')!= -1:#Green
            Pch3 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            GR = np.float64(Pch3)  
    
    #Generate a RGB image from red-, green-, and blue- grascale images
    RGB = cv2.merge((BL,GR,RD))
    cv2.imwrite(drct+timestamp+"_RGB.png",RGB)
    RGB_adj = RGB*2
    cv2.imwrite(drct+timestamp+"_RGBadj.png",RGB_adj)
    
    #Calculate Anthocyanin content index; these calculation need float64 format
    with np.errstate(divide='ignore'):# to ignore zero values of denominator during the index calculation
    #ACI calculation (R_Red - R_Green)/(R_Red + R_Green), where R is reflectance value extracted from the multipsectral images
        ACI_before_masking = np.true_divide(np.subtract(RD,GR), np.add(RD,GR))+1#to have unique values between the object and background 
    #remove the background by conjugating foreground (ACI image) and the background (binary image for object of interest) 
    ACI = cv2.bitwise_and(ACI_before_masking,ACI_before_masking, mask=masking)#mask contains 1 or 0, where 1 is the object of interet and 0 is background
    ACI = np.where(ACI == 0, -1, ACI)#replace the background value, 0, as -1 to have better contrast between background and the object of interest
    ACI = np.where(ACI != -1, ACI-1, ACI)#after the separation between the object and background, subtract 1 again to have the orginial index value
        
    #Calculate Normalized difference vegetation index
    with np.errstate(divide='ignore'):# to ignore zero values of denominator during the index calculation
    #NDVI calculation (R_IFR - R_Red)/(R_IFR + R_Red), where R is reflectance value extracted from the multipsectral images
        NDVI_before_masking = np.true_divide(np.subtract(IFR,RD), np.add(IFR,RD))+1#to have unique values between the object and background 
    #remove the background by conjugating foreground (NDVI image) and the background (binary image for object of interest) 
    NDVI = cv2.bitwise_and(NDVI_before_masking,NDVI_before_masking, mask=masking)#mask contains 1 or 0, where 1 is the object of interet and 0 is background
    NDVI = np.where(NDVI == 0, -1, NDVI)#replace the background value, 0, as -1 to have better contrast between background and the object of interest
    NDVI = np.where(NDVI != -1, NDVI-1, NDVI)#after the separation between the object and background, subtract 1 again to have the orginial index value

    # Plot a picture of ACI
    plt.colorbar(plt.imshow(ACI), fraction = 0.046, pad=0.04)# to generate a color scale bar
    plt.clim(-1,1)#range of the color gradient for normalized index image
    plt.axis('off') #Not to show the axis info of the image
    plt.savefig(drct+timestamp+"_ACI.jpg", dpi = 300) #save the ACI image in the directory of where the image is located
    plt.close() #close the ACI image
        
    # Plot a picture of NDVI
    plt.colorbar(plt.imshow(NDVI), fraction = 0.046, pad=0.04)# to generate a color scale bar
    plt.clim(-1,1)#range of the color gradient for normalized index image
    plt.axis('off') #Not to show the axis info of the image
    plt.savefig(drct+timestamp+"_NDVI.jpg", dpi = 300) #save the NDVI image in the directory of where the image is located
    plt.close() #close the NDVI image
            
    #Plot a histogram of ACI
    ACI_Hist = ACI[ACI !=-1]#remove the background, which has value of -1
    weights = np.ones_like(ACI_Hist)/float(len(ACI_Hist))
    plt.hist(ACI_Hist, weights=weights, bins=49, rwidth =0.85) #ADJUSTED BINS TO 49 NTO GET A SMOOTEHR HISTOGRAM
    plt.xlabel('ACI')#put a label on x-axis
    plt.ylabel('Proportion')#put a label on y-axis
    plt.title('Normalized ACI') #title of the histogram
    plt.savefig(drct+timestamp+"_ACI_HIST.jpg", dpi = 300) #save the Histogram in the directory of where the image is located
    plt.close() #close the histogram
    Avg_ACI = np.mean(ACI_Hist) #averaged value from the histogram
    Std_ACI = np.std(ACI_Hist) #standard deviation from the histogram
    Avg_ACI3 = np.around(Avg_ACI, decimals=3) #round to 3 decimals
    Std_ACI3 = np.around(Std_ACI, decimals=3)
    

    #Plot a histogram of NDVI
    NDVI_Hist = NDVI[NDVI !=-1]#remove the background, which has value of -1
    weights = np.ones_like(NDVI_Hist)/float(len(NDVI_Hist))
    plt.hist(NDVI_Hist, weights=weights, bins=300, rwidth =0.85)
    plt.xlim(-0.3,1)
    plt.ylim(0,0.15)
    plt.xlabel('NDVI')#put a label on x-axis
    plt.ylabel('Proportion')#put a label on y-axis
    plt.title('Normalized NDVI') #title of the histogram
    plt.savefig(drct+timestamp+"_NDVI_HIST.jpg", dpi = 300) #save the Histogram in the directory of where the image is located
    plt.close() #close the histogram
    Avg_NDVI = np.mean(NDVI_Hist) #averaged value from the histogram
    Std_NDVI = np.std(NDVI_Hist) #standard deviation from the histogram
    Avg_NDVI3 = np.around(Avg_NDVI, decimals=3) #round to 3 decimals
    Std_NDVI3 = np.around(Std_NDVI, decimals=3)
    

    #write the filename, averaged value of histogram, std value of histogram and PCS into a given csv file name
    with open(csvname, "a", newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow([filename, Avg_ACI3, Std_ACI3, Avg_NDVI3, Std_NDVI3, cm_area, PCS, Thr])
    # IF THE ABOVE LINE GIVES AN ERROR MESSAGE, REPLACE WRITEROW WITH WRITE. THE CORRECT FORMAT APPEARS TO DEPENDS ON THE VERSION OF THE OpenCV package
    csvfile.close()
    
    #opens images for verification
    test = cv2.imread(drct+timestamp+"_NDVI.jpg")
    cv2.imshow("Img", test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    test2 = cv2.imread(drct+timestamp+"_RGBadj.png")
    cv2.imshow("Img", test2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
           
#IEMI instruction sets minimum threshold value and designates location to save output csv file
# Minimum pixel size, maximum pixel size, cminimium threshold, File name time stamp and name, file format folder 
#Normal
IEMI(50, 90000000, 6, "/home/pi/Desktop/MultiSpectralImaging/Kahlin/pHv3/"+Date+"/"+Date+"_Pigment_index_HIST.csv",'/*.jpg',directory)

#Size Calibration
#IEMI(50, 90000000, 1, "/home/pi/Desktop/MultiSpectralImaging/Kahlin/pHv3/SizeCal/"+Date+"_sizecal.csv",'/*.jpg',directory)



print("Finished. Images saved in 'Pictures' folder in a subfolder with date and time stamp.")

camera.init_camera() #turn on the camera again to reset after long exposure time for mask image
time.sleep(0.1)
camera.capture(encoding = 'jpeg')
time.sleep(0.1)
camera.close_camera()
#calibration (Jun 13)


