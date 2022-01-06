from datetime import datetime
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

#define video
#variable = cv2.VideoCapture(definedvideo)

#frames = 0
#while(variable.isOpened()):


#read frame by frame{

#success,image = variable.read()
#define variables
#count = 0
#frames =1
#skip_frames = 30
#loop function for using only frames % 30
#while success:
#   ret,frame= variable.read()
#       if ret and frames % skip_frames == 0
# save frame as JPEG file
#           cv2.imwrite("frame%d.jpg" % count, image)
#           success,image = variable.read()
#add plus one to variables to use another frame and save next image with different name
#           count += 1
#           frames += 1
#   else
#       frames+=1

#}


#save image to variable
#image = pil.Image.open('image')

#find a way to save only needed information






#Read in Image, Grayscale and Blur
#img = cv2.imread('/homejouadamis/PycharmProjects/pythonProject/image%d.jpg' %count)

#define variable img for used image
img = cv2.imread('/home/jouadamis/PycharmProjects/pythonProject/image1.jpg')
#converting picture to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Apply filter and find edges for localization

#using bilateral Filter that reduces noise
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
#detecting edges
edged = cv2.Canny(bfilter, 30, 200)

#Find Contours and Apply Mask

#find edges
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#simplifying return of contours
contours = imutils.grab_contours(keypoints)
#return of top ten kontours
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#filtring(looping) thrue contours to find number plate shape(rectangle)
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

#
mask = np.zeros(gray.shape, np.uint8)
#finding numberplate 
new_image = cv2.drawContours(mask, [location], 0,255, -1)
#overlaying everything except numberplate
new_image = cv2.bitwise_and(img, img, mask=mask)

#storing coordinates in variables x,y
(x,y) = np.where(mask==255)
#storing top left point(min) and bottom right point(max)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
#getting variable croped_image
cropped_image = gray[x1:x2+1, y1:y2+1]


#Use EasyOCR To Read Text
#define in what leanguage to read in
reader = easyocr.Reader(['en'])
#read text from variable cropped_image
result = reader.readtext(cropped_image)

#Render Result
text1 = result[0][-2]
text2 = result[1][-2]
text = text1 + " " + text2
#define font on picture
font = cv2.FONT_HERSHEY_SIMPLEX
#
res = cv2.putText(img, text=text , org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
#open txt file
f = open("demofile.txt", "a")
#define date + time to variable time
time = datetime.now()
#write in file SPZ and date+time
f.write(text +"    " + str(time) + " \n")
#close file
f.close()

res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
#plt.imsave('image_new%d.jpg', % count ,  cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
#save image
plt.imsave('image_new1.jpg', cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
