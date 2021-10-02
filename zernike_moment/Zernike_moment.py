import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dlib
import pandas as pd 
import numpy as np 
from scipy.spatial import distance as dist
from tqdm.notebook import tqdm
import os
from sklearn.preprocessing import OneHotEncoder
# import the necessary packages

import imutils
from pyzernikemoment import Zernikemoment



if __name__ == '__main__':
    sayac = 0
    veri=np.array([])
    sutun_isimleri = ['mouth_g', 'mouth_f',"righteye_g","righteye_f","lefteye_g","lefteye_f","nose_g","nose_f","Sonuc_Alert","Sonuc_tired"]
    
    img_path="C:/Users/oguzh/Desktop/kod/ouz"
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
    
    train_paths , train_labels = setup(img_path)
    
    print(train_paths[0])
    
    
    liste=[]
    
    for i in train_paths:
      
        imgs=cv2.imread(i)
        img = imutils.resize(imgs, width=500)       
        imgs=(cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB))
        images = cv2.imread("beyaz.jpg")
        images = imutils.resize(images, width=600)
        plt.imshow(imgs)
       
    
       
       
    
       
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
       
       
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_frame, 0)
        def visualize_facial_landmarks(image, shape,j,k,images, colors=None, alpha=1):
  
            overlay = image.copy()
            output = image.copy()
            outputs = images.copy()
            overlays = images.copy()
            
            colors = [(0, 0, 0)]
  
            
            pts = shape[j:k]
   
            hull = cv2.convexHull(pts)
            
            cv2.drawContours(overlay, [hull], -1, colors[0], -1)
            cv2.drawContours(overlays, [hull], -1, colors[0], -1)
            
            cv2.addWeighted(overlay, alpha, output,0, 0, output)
            cv2.addWeighted(overlays, alpha, outputs,0, 0, outputs)
            plt.imshow(overlays)
            
   
            return outputs
    
            
           
           
        def shape_to_np(shape, dtype="int"):
            # (x, y) koordinatlarinin listesini baslat
            coords = np.zeros((68, 2), dtype=dtype)
       
            # 68 yuz simgesi uzerinde dongu
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
       
       
            return coords

         
    
       
        if len(rects) > 0:
            for rect in rects:
                shape = predictor(gray_frame, rect)
                shape = shape_to_np(shape)
               
                    #nereyi istiyorsak oraya karsilik gelen koordinatlar
                
                n = 4 #radyal derece    
                m = 2 #m, "azimutal" veya "açısal frekans
                       
                
                output_1 = visualize_facial_landmarks(imgs, shape,48,68,images)
                
              
                output_1 = cv2.cvtColor(output_1, cv2.COLOR_BGR2GRAY)
                Z_1, A_1, Phi_1 = Zernikemoment(output_1, n, m)
                mouth_g = str(round(A_1, 4))
                mouth_f = str(round(Phi_1, 4))
                
                output_2 = visualize_facial_landmarks(imgs, shape,36,42,images)
                
                output_2 = cv2.cvtColor(output_2, cv2.COLOR_BGR2GRAY)
                Z_2, A_2, Phi_2 = Zernikemoment(output_2, n, m)
                righteye_g = str(round(A_2, 4))
                righteye_f = str(round(Phi_2, 4))
                
                output_3 = visualize_facial_landmarks(imgs, shape,42,48,images)
               
                output_3 = cv2.cvtColor(output_3, cv2.COLOR_BGR2GRAY)
                Z_3, A_3, Phi_3 = Zernikemoment(output_3, n, m)
                lefteye_g = str(round(A_3, 4))
                lefteye_f = str(round(Phi_3, 4))
                
                output_4 = visualize_facial_landmarks(imgs, shape,27,35,images)
               
                output_4 = cv2.cvtColor(output_4, cv2.COLOR_BGR2GRAY)
                Z_4, A_4, Phi_4 = Zernikemoment(output_4, n, m)
                nose_g = str(round(A_4, 4))
                nose_f = str(round(Phi_4, 4))
                
                #print(str(round(A, 4)))
                #print(str(round(Phi, 4)))
                
                
               
        else:
            sayac = sayac + 1
            print(sayac)
            print("b")
           
            mouth_g = 0.0
            mouth_f = 0.0
            righteye_g = 0.0
            righteye_f = 0.0
            lefteye_g = 0.0
            lefteye_f = 0.0
            nose_g = 0.0
            nose_f = 0.0
        a=[[mouth_g,mouth_f,righteye_g,righteye_f,lefteye_g,lefteye_f,nose_g,nose_f]]
        liste.extend(a)
                   
    
    
    liste = np.reshape(liste,(4,8))
    #print(liste)
    
    
    
    
    csv=np.concatenate([liste,train_labels],axis=1)
    
    
    
    veriseti = pd.DataFrame(data=csv, columns=sutun_isimleri)
    #print(veriseti)
    
    veriseti.to_csv("ouz1.csv",index=False)