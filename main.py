from datetime import datetime
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

#define video
video = cv2.VideoCapture('/home/jouadamis/PycharmProjects/pythonProject/video1.mp4')
#frames = 0
#while(video.isOpened()):
#read frame by frame
success,image = video.read()
#define variables
count = 0
frames = 1
skip_frames = 29
#loop function for using only frames % 31
while success:
    ret, frame = video.read()
    if ret and frames % skip_frames == 0:
        # save frame as JPEG file
        cv2.imwrite("frame%d.jpg" % count, image)
        #my trie to save image as variable not a .jpg
        #img = plt.Image.open(image)
        success,image = video.read()
        #add plus one to variables to use another frame and save next image with different name
        count += 1
        frames += 1

    else:
        frames += 1

#save image to variable
#image = pil.Image.open('image')
#loop whole code to read every saved frame
number = 0
for number in range(count):
    #define variable img for used image
    img = cv2.imread('/home/jouadamis/PycharmProjects/pythonProject/frame%d.jpg' %number)
    #img = cv2.imread('/home/jouadamis/PycharmProjects/pythonProject/image1.jpg')
    #converting picture to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #using bilateral Filter that reduces noise
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    #detecting edges
    edged = cv2.Canny(bfilter, 30, 200)
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
    #define in what leanguage to read in
    reader = easyocr.Reader(['en'])
    #read text from variable cropped_image
    result = reader.readtext(cropped_image)
    f = open("file.txt", "a")
    f.write(str(result)  + " \n")
    f.close()
    #Render Result
    text1 = result[0][-2]
    text2 = result[1][-2]
    text = text1 + " " + text2
    #define font on picture
    font = cv2.FONT_HERSHEY_SIMPLEX
    #putting text in picture
    res = cv2.putText(img, text=text , org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    #open txt file
    f = open("demofile.txt", "a")
    #define date + time to variable time
    time = datetime.now()
    #write in file SPZ and date+time
    f.write(text +"    " + str(time) + " \n")
    #close file
    f.close()
    #drawing rectangle in picture
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
    #save image
    plt.imsave('/home/jouadamis/PycharmProjects/pythonProject/image_new%d.jpg' %number ,  cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    #plt.imsave('image_new1.jpg', cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
