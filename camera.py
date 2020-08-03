import cv2
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6
from PIL import Image
from numpy import asarray
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    # def get_embedding(self,model, face_pixels):
    #     # scale pixel values
    #     face_pixels = face_pixels.astype('float32')
    #     # standardize pixel values across channels (global)
    #     mean, std = face_pixels.mean(), face_pixels.std()
    #     face_pixels = (face_pixels - mean) / std
    #     # transform face into one sample
    #     samples = np.expand_dims(face_pixels, axis=0)
    #     # make prediction to get embedding
    #     yhat = model.predict(samples)
    #     return yhat[0]
    
    def get_frame(self):
        success, image = self.video.read()

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)

        temp = image
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        margin = 10
        for (x,y,w,h) in face_rects:
        	# cv2.rectangle(image,(x-margin,y-margin),(x+w+margin,y+h+margin),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)




        for (x,y,w,h) in face_rects:
            print(x,y,w,h)
            temp = temp[y-margin:y+h+margin,x-margin:x+w+margin]
            
            # temp = temp[0:200,x:w+x]
            # temp = cv2.resize(temp,(160,160))
            # temp2 = Image.fromarray(temp,'RGB')
            # facearrray = asarray(temp)
            # emb = self.get_embedding(facenet_model,facearrray)
            ret,face = cv2.imencode('.jpg',temp)
            return face.tobytes()

        # temp = 


















        return jpeg.tobytes()
