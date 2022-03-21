# import the opencv library
import cv2
import os

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frames = vid.read()

    gray = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Overall circle
        cv2.circle(frames, (x + (int)(w / 2), y + (int)(h / 2)), ((int)(w / 2)), (255, 0, 0), -1)
        # Two eyes
        cv2.circle(frames, (x + int(w / 3), (y + int(5*h/ 16))), ((int)(w/9)), (0, 255, 0), -1)
        cv2.circle(frames, (x + int(2*w / 3), (y + int(5 * h / 16))), ((int)(w / 9)), (0, 255, 0), -1)

        # Smile
        cv2.ellipse(frames, (x+(int)(w/2), y+(int)(11*h/16)), ((int)(w/4), (int)(h/8)), 0, 0, 180, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow('frame', frames)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()