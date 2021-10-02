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
from collections import OrderedDict
import imutils
from pyzernikemoment import Zernikemoment



if __name__ == '__main__':
    sayac = 0
    veri=np.array([])
    sutun_isimleri = ['rightEAR', 'leftEAR',"avgEAR","noseEAR","mouthEAR","gen","faz","Sonuc_Alert","Sonuc_tired"]
    
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
        img=mpimg.imread(i)
        img=(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imgs=cv2.imread(i)
        
        
            
            
        imgs=(cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB))
            
            
        images = cv2.imread("beyaz.jpg")
        images = imutils.resize(images, width=600)
        plt.imshow(imgs)
        
            
            # define a dictionary that maps the indexes of the facial
            # landmarks to specific face regions
      
        
       
    
       
       
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
       
       
       
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
       
       
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_frame, 0)
        def visualize_facial_landmarks( shape,images, alpha=1):
       
            
            
            outputs = images.copy()
            
            
            colors = [(0, 0, 0)]
  
            (j, k) = (42, 48)
            pts = shape[j:k]
   
            hull = cv2.convexHull(pts)
            
            
            cv2.drawContours(outputs, [hull], -1, colors[0], -1)
            
            plt.imshow(outputs)
            
            
   
            return outputs
         
    
       
        if len(rects) > 0:
            for rect in rects:
                shape = predictor(gray_frame, rect)
                shape = shape_to_np(shape)
               
                    #nereyi istiyorsak oraya karsilik gelen koordinatlar
                
                mouth=shape [48:68]
                left_eye = shape[42: 48]
                right_eye = shape[36: 42]
                nose=shape[27:35]
                
                sayac = sayac + 1
                n = 4
                m = 0
                print(sayac)       
                leftEAR = goz_enboy_orani(left_eye)
                rightEAR = goz_enboy_orani(right_eye)
                mouthEAR=agiz_acikligi(mouth)
                noseEAR=burun_seviyesi(nose)
                avgEAR = (leftEAR + rightEAR) / 2.0
                output = visualize_facial_landmarks(shape,img)
                cv2.imwrite("zz.jpg",output)
                a = cv2.imread("zz.jpg")
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                Z, A, Phi = Zernikemoment(a, n, m)
                ZZ = str(round(A, 4))
                AA = str(round(Phi, 4))
                print(Z)
                gen = ZZ
                faz = AA
                #print(str(round(A, 4)))
                #print(str(round(Phi, 4)))
                
                
               
        else:
            sayac = sayac + 1
            print(sayac)
            print("b")
            leftEAR = 0.0
            rightEAR =0.0
            mouthEAR=0.0
            noseEAR=0.0
            avgEAR = 0.0
            gen = 0.0
            faz = 0.0
        a=[[rightEAR,leftEAR,avgEAR,noseEAR,mouthEAR,gen,faz]]
        liste.extend(a)
                   
    
    
    liste = np.reshape(liste,(4,7))
    #print(liste)
    
    
    
    
    csv=np.concatenate([liste,train_labels],axis=1)
    
    
    
    veriseti = pd.DataFrame(data=csv, columns=sutun_isimleri)
    #print(veriseti)
    
    veriseti.to_csv("train2.csv",index=False)