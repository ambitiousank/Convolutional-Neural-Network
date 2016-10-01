import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm


feature_horse=np.load("HorseFeature.npy")
feature_bike=np.load("BikeFeature.npy")
label_bike=np.load("BikeLabel.npy")
label_horse=np.load("HorseLabel.npy")



#Splitting Train and Test Data

train=(len(feature_horse)/10)*7
#print train
#print len(feature_horse)


X_train=feature_horse[:train]
#print X_train.shape
Y_train=label_horse[:train]
X_test=feature_horse[train:]
Y_test=label_horse[train:]

train=(len(feature_bike)/10)*7
#print train
#print len(feature_bike)

X_train=np.append(X_train,feature_bike[:train],axis=0)
Y_train=np.append(Y_train,label_bike[:train],axis=0)
X_test=np.append(X_test,feature_bike[train:],axis=0)
Y_test=np.append(Y_test,label_bike[train:],axis=0)
#print feature_bike[:train].shape

#print X_train.shape
#print Y_train

model=LogisticRegression()
#model=svm.SVC()
model.fit(X_train,Y_train)
predictions=model.predict(X_test)

print "accuracy for ensemble ",metrics.accuracy_score(Y_test,predictions)
print "f1 score for ensemble ",metrics.f1_score(Y_test,predictions)

print predictions

print Y_test

#print len(X_train)
#print len(Y_train)
#print len(X_test)
#print len(X_test)

