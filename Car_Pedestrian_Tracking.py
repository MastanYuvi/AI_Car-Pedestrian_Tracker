
import cv2

#Our Car Image
#img_file = "car_image.jpg"
video = cv2.VideoCapture('Ped_Shorted.mp4')

#Our pre trained car Classifier
Classifier_file = 'car_detector.xml'

Ped_Classifier = 'pedestrian_detector.xml'

#Classifier Files
car_tracker = cv2.CascadeClassifier(Classifier_file)

ped_tracker = cv2.CascadeClassifier(Ped_Classifier)

#Run through every frame of Video
while True:
    #read every frame from video
    (read_successful, frame) = video.read()
    #safe_coding
    if read_successful:

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        break

    cars = car_tracker.detectMultiScale(grayscale_frame)

    peds = ped_tracker.detectMultiScale(grayscale_frame)

    for (x, y, w, h) in cars:            
        cv2.rectangle(frame, (x+1,y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in peds:            
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
    
    #Display the image with Spotted Objects
    cv2.imshow('AI Car Detector', frame)

    #Dont autoclose() wait for keypress
    key = cv2.waitKey(1)

    #stop the code by pressing Q button
    if key==81 or key==113:
        break



    print("Code Completed")

video.release()

"""
#pedestrian_detector.xml
#create open CV Image
img = cv2.imread(img_file)

#Convert to Grayscale (neede for haar Cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create Car classifier
car_tracker = cv2.CascadeClassifier(Classifier_file)

#Detect Cars once u have a Classifier Object and we will get Coordinated of Detected cars in Rectangles [388 298 890 567] etc
cars = car_tracker.detectMultiScale(black_n_white)

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)

#Display the image with Spotted Objects
cv2.imshow('AI Car Detector', img)

#Dont autoclose() wait for keypress
cv2.waitKey()

print("Code Completed")
"""