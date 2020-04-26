import cv2

scale_factor = 1.3
min_neighbors = 3
min_size = (100, 100)
webcam=False
 
def detect(path):
 
    cascade = cv2.CascadeClassifier(path)
    video_cap = cv2.VideoCapture("C:/Users/iamvr/Desktop/Dataset/videoplayback.mp4")
    i=0
    while True:
        # Capture frame-by-frame
        ret, img = video_cap.read()
 
        #converting to gray image for faster video processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        rects = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                         minSize=min_size)
        # if at least 1 face detected
        if len(rects) >= 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in rects:
                face = gray[y:y+h,x:x+w]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite('C:/Users/iamvr/Desktop/Dataset/'+str(i)+'.jpg',face)
                i+=1
            # Display the resulting frame
            cv2.imshow('Face Detection on Video', img)
            
            #wait for 'c' to close the application
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
    video_cap.release()
 
def main():
    cascadeFilePath="C:/Users/iamvr/Desktop/Dataset/haarcascade.xml"
    detect(cascadeFilePath)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()