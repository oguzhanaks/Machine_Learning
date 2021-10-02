from PIL import Image   #resimleri preprocess yaparken kullanacağız.
import matplotlib.pyplot as plt 
import numpy as np 
import os #resimleri klasörden çekerken ,import edekerden kullanacağız
import time #algoritmanın koşum zamanınnı ölçer
#from tkinter import *
import seaborn as sns
plt.style.use('seaborn-whitegrid')   #grafik tarzı kareli olur
import pandas as pd 


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

data=pd.read_csv('C:/Users/oguzh/Desktop/kod/trainn2.csv',sep=',')
test=pd.read_csv('C:/Users/oguzh/Desktop/kod/testt2.csv',sep=',')



#tek kolonda class değerini tutuyoruz.
data['class']=[0 if i==1 else 1 for i in data['Sonuc_Alert']]
test['class']=[0 if i==1 else 1 for i in test['Sonuc_Alert']]





#Geçersiz verileri silme:
x=data[data.gen_1==0.0000].index
print("Geçersiz indexler:",x)
print("Geçersiz index sayısı:",len(x))
data=data.drop(x,axis=0).reset_index(drop=True)

x=test[test.gen_1==0.0000].index
test=test.drop(x,axis=0).reset_index(drop=True)






#train-test split'a hazırlama:
y=data['class'].values
x=data.drop(labels=['class','Sonuc_Alert','Sonuc_tired'],axis=1)

test_y=test['class'].values
test_x=test.drop(labels=['class','Sonuc_Alert','Sonuc_tired'],axis=1)



#train-test split:
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.33,random_state=50)




dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}   #Ağaç derinliği:Ağacın maksimum derinliği. Yok ise, düğümler tüm yapraklar saf olana kadar veya tüm yapraklar min_samples_split'den daha az örnek içerene kadar genişletilir.
                #Genel olarak, ağacınızın ne kadar derin büyümesine izin verirseniz, modeliniz o kadar karmaşık hale gelir çünkü daha fazla bölünmeye sahip olursunuz ve veriler hakkında daha fazla bilgi toplar ve bu,
                # karar ağaçlarına aşırı uyum sağlamanın temel nedenlerinden biridir çünkü modeliniz eğitim verileri için mükemmel uyum sağlar(overfitting) ve test seti üzerinde iyi bir genelleme yapamaz. Dolayısıyla,
                # modeliniz gereğinden fazla uyuyorsa, max_depth sayısını azaltmak, aşırı uydurmayla mücadele etmenin bir yoludur.

                #Genelde yaptığım şey, modelin önce max_depth'e karar vermesine izin vermek ve ardından tren ve test puanlarımı karşılaştırarak 
                # aşırı uygunluk(overfitting) veya yetersiz uygunluk(underfitting) ararım ve maksimum derinliği düşürdüğüm veya artırdığım dereceye bağlı olarak.

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000],
                  "probability":[True]}

rf_param_grid = {"min_samples_split":[2,3,10,15,14,13],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,500],      #Daha fazla ağaç sayısı size daha iyi performans sağlar ancak kodunuzu yavaşlatır. 
                                               #İşlemcinizin kaldırabileceği kadar yüksek değeri seçmelisiniz çünkü bu, tahminlerinizi daha güçlü ve daha kararlı hale getirir.

                "criterion":["gini"]}         #criterion:Bir bölünmenin kalitesini ölçme işlevidir. 1-gini:Safsızlık( bir düğüm üzerindeki etiketlerin homojenliğinin bir ölçüsüdür.)     2-entropy:bilgi kazanımı için Bilgi kazanımı, kirlilik ölçüsü olarak entropi ölçüsünü kullanır ve bir düğümü, en fazla bilgi kazancı sağlayacak şekilde böler. Gini Safsızlık, hedef özniteliğin değerlerinin olasılık dağılımları arasındaki farklılıkları ölçer ve bir düğümü en az miktarda kirlilik verecek şekilde böler. ÖNEMLİ NOT:Araştırmacıların çoğu, çoğu durumda, ayırma kriterlerinin seçiminin ağaç performansında çok fazla fark yaratmayacağına işaret ediyor. bu iki durum için şu söylenebilir:hemen hemen her ikisini de kullanın, ancak tek fark entropinin hesaplanması biraz daha yavaş olabilir çünkü logaritmik bir işlevi hesaplamanızı gerektirir


logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"], 
                    "solver" : ["liblinear"]}


knn_param_grid = {"n_neighbors": np.linspace(1,13,13, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]



random_state = 42
classifier =[DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv= StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
 #n_jobs=-1 yapmak kodu paralel koşar.Tüm işlemcileri kulllanır. verbose=1 yaptığımızda kod koşarken sonuçları gösterir
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


cv_results=pd.DataFrame({'Cross Validation Means':cv_result,'ML Models':["DecisionTreeClassifier",
             "SVC",
             "RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g=sns.barplot('Cross Validation Means','ML Models',data=cv_results)
g.set_xlabel('Mean Accuracy')
g.set_title('Cross Validation Score')
plt.show()

votingC=VotingClassifier(estimators=[('dt',best_estimators[0]),
                                      ('svc',best_estimators[1]),
                                      ('rf',best_estimators[2])],
                                      voting='soft',n_jobs=-1)
            #vorting=hard olsaydı en çok sayıya sahip değer dönerdi. Sorf ise istatiksel bir hesaplama yaparak değer döndürür.




votingC=votingC.fit(x_train,y_train)
print("VAL:accuracy_score",accuracy_score(votingC.predict(x_val),y_val))




print("TEST:accuracy_score",accuracy_score(votingC.predict(test_x),test_y))