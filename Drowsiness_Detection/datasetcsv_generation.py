import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dlib
import pandas as pd 
import numpy as np 
from scipy.spatial import distance as dist
from tqdm.notebook import tqdm
import os
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dlib
import pandas as pd
import numpy as np
from scipy.spatial import distance as dist
from tqdm.notebook import tqdm
import os
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from imutils import face_utils
import imutils
from collections import OrderedDict 
from PIL import ImageChops
import cv2
import matplotlib.pylab as plt
from pyzernikemoment import Zernikemoment


import os
import glob,os
from glob import glob
if __name__ == '__main__':
    
    
    veri=np.array([])
    sutun_isimleri = ['rightEAR', 'leftEAR',"avgEAR","noseEAR","mouthEAR","Sonuc_Alert","Sonuc_tired"]
    
    resim="C:/Users/oguzh/Desktop/kod/ouz"
    def setup(base):
        paths = []
        labels = []
        states = ['alert','tired']
        for label in tqdm(states):
            temp_base = base + f'/{label}'
            for img in os.listdir(temp_base):
                paths.append(temp_base + f'/{img}')
                labels.append(label)
                
        enc = OneHotEncoder(sparse = False)
        labels = np.reshape(labels, (-1, 1))
        enc.fit(labels)
        labels = enc.transform(labels)
        
        return np.array(paths), labels
    
    train_paths , train_labels = setup(resim)
    
    print(train_paths[0])
    
    
    liste=[]
    
    for i in train_paths:
            
        img=mpimg.imread(i)
        imgs=cv2.imread(i)
        
        img=(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imgs=(cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB))
        
        
        images = cv2.imread("beyaz.jpg")
        images = imutils.resize(images, width=500)
        plt.imshow(imgs)
        facial_features_cordinates = {}
        
        # define a dictionary that maps the indexes of the facial
        # landmarks to specific face regions
        FACIAL_LANDMARKS_INDEXES = OrderedDict([
            ("Mouth", (48, 68)),
          
            ("Right_Eye", (36, 42)),
            ("Left_Eye", (42, 48)),
            ("Nose", (27, 35)),
            
        ])
    
        
        
        def goz_enboy_orani(eye):
            #goz-enboy_orani olarak tanımlanan fonksiyonuna parametre olarak yüz üzerindeki göze
            # ait noktaların atandığı dizi olan eye dizisi gönderilir.
            #‘a’ ile tanımlanan kısımda eye[1],eye[5] yani yüz üzerindeki 38-42. noktalar arasındaki
            # öklid uzaklığı ve ‘b’ ile de 39-41 arasındaki mesafe hesaplanır.
            #Daha sonra ‘c’ de 37-40 arasındaki yatay mesafe hesaplanarak ear ile tanımlanan değişkende bu değerler işleme alınır.
        
            # oklid mesafesini hesapliyorum
            #gozun koordinatlari
            a = dist.euclidean(eye[1], eye[5])
            b = dist.euclidean(eye[2], eye[4])
        
            # yataydaki oklid mesafesi
        
            c = dist.euclidean(eye[0], eye[3])
        
            ear = (a + b) / (2 * c)
        
            return ear
        def agiz_acikligi(mouth):
            #Ağız içinde aynı mantıkta ilerleyerek ağız açıklığını bulmak
            # için en üst ve en alt nokta arasındaki uzaklık bulunur.
            a = dist.euclidean(mouth[4], mouth[10])
            mouth=a
        
            return mouth
        
        def burun_seviyesi(nose): # oklid mesafesini hesapliyorum
            #gozun koordinatlari
            a = dist.euclidean(nose[0],((right_eye[0]+left_eye[0])/2))
            nose=a
            #print(nose)
        
            return nose
        def shape_to_np(shape, dtype="int"):
            # (x, y) koordinatlarinin listesini baslat
            coords = np.zeros((68, 2), dtype=dtype)
        
            # 68 yuz simgesi uzerinde dongu
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        
        
            return coords
        def visualize_facial_landmarks(image, shape,images, colors=None, alpha=1):
            # create two copies of the input image -- one for the
            # overlay and one for the final output image
            overlay = image.copy()
            output = image.copy()
            outputs = images.copy()
            overlays = images.copy()
            
        
            # if the colors list is None, initialize it with a unique
            # color for each facial landmark region
            if colors is None:
                colors = [(0, 0, 0), (0, 0, 0),(0, 0, 0),
                          (0, 0, 0),(0, 0, 0),
                          (0, 0, 0), (0, 0, 0)]
        
            # loop over the facial landmark regions individually
            for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
                # grab the (x, y)-coordinates associated with the
                # face landmark
                (j, k) = FACIAL_LANDMARKS_INDEXES[name]
                pts = shape[j:k]
                facial_features_cordinates[name] = pts
        
                # check if are supposed to draw the jawline
                if name == "Jaw":
                    # since the jawline is a non-enclosed facial region,
                    # just draw lines between the (x, y)-coordinates
                    for l in range(1, len(pts)):
                        ptA = tuple(pts[l - 1])
                        ptB = tuple(pts[l])
                  
                        cv2.line(overlay, ptA, ptB, colors[i], 2)
                        cv2.line(overlays, ptA, ptB, colors[i], 2)
                        
                        
                        
                        
                # otherwise, compute the convex hull of the facial
                # landmark coordinates points and display it
                else:
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(overlay, [hull], -1, colors[i], -1)
                    cv2.drawContours(overlays, [hull], -1, colors[i], -1)
                    
        
            # apply the transparent overlay
            plt.imshow(output)
            plt.imshow(overlay)
            plt.imshow(overlays)
            cv2.addWeighted(overlay, alpha, output,0, 0, output)
            cv2.addWeighted(overlays, alpha, outputs,0, 0, outputs)
            
        
            # return the output image
            
            return output
    
    
        
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_frame, 1)
      
    
        
        if len(rects) > 0:
            for rect in rects:
                print("zzazaza")
                shape = predictor(gray_frame, rect)
                shape = shape_to_np(shape)
                output = visualize_facial_landmarks(imgs, shape,imgs)
                cv2.imwrite("zz.jpg",output)
                
                
                mouth=shape [48:68]
                left_eye = shape[42: 48]
                right_eye = shape[36: 42]
                nose=shape[27:35]
                       
                    
             
                leftEAR = goz_enboy_orani(left_eye)
                rightEAR = goz_enboy_orani(right_eye)
                mouthEAR=agiz_acikligi(mouth)
                noseEAR=burun_seviyesi(nose)
                avgEAR = (leftEAR + rightEAR) / 2.0
                
                
               
                
                
                
        else:
         
            leftEAR = 0.0
            rightEAR =0.0
            mouthEAR=0.0
            noseEAR=0.0
            avgEAR = 0.0
         
            
        a=[[rightEAR,leftEAR,avgEAR,noseEAR,mouthEAR]]
        liste.extend(a)
                    
           
            
            
               
         
    
    
    
    liste = np.reshape(liste,(4,5))
    #print(liste)
    
    
    
    
    csv=np.concatenate([liste,train_labels],axis=1)
    
    
    
    veriseti = pd.DataFrame(data=csv, columns=sutun_isimleri)
    #print(veriseti)
    
    veriseti.to_csv("train.csv",index=False)
    
