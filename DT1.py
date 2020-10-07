import sklearn
from sklearn import tree
import numpy as np
import pandas
from sklearn.feature_selection import  VarianceThreshold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import pickle as pkl
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import time

df = pandas.read_csv('test_set.csv')
test = pandas.read_csv('new_data.csv')
testset = test.values
dataset = df.values

a = dataset[:,4]
d = dataset[:,0:4]
c = dataset[:,5:21]
b = dataset[:,0:21]
b = np.delete(b,4,axis = 1)

a1 = testset[:,4]
d1 = testset[:,0:4]
c1 = testset[:,5:21]
b1 = testset[:,0:21]
b1 = np.delete(b1,4,axis = 1)

#b = a+c
#b = np.vstack((d,c))
#b = np.append(d,c) 
#print(c.shape)
#print(d.shape)
#print(b.shape)

################################################################################
x_train,x_test,y_train,y_test=train_test_split(b,a,test_size=0.3)

start = time.time()

clf=DecisionTreeClassifier(criterion='gini',splitter='random',max_depth=20)
clf.fit(x_train,y_train)
predict_target=clf.predict(x_test)
score = clf.score(x_test,y_test)
print('The accuacy of training with DT is %f',score)

test_target = clf.predict(b1)
score1 = clf.score(b1,a1)
print('The accuacy of test with DT is %f',score1)

end = time.time()
print (end-start)

###########################################################################################
x_train,x_test,y_train,y_test=train_test_split(c,a,test_size=0.3)

start = time.time()

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
print('The accuacy of training with KNN is %f',knn.score(x_test, y_test))
knn.fit(c1,a1)
print('The accuacy of test with KNN is %f',knn.score(c1, a1))

end = time.time()
print (end-start)
#print(knn.score(c1, a1))

###########################################################################################
voting_clf2 = VotingClassifier(estimators=[
    ('knn_clf',KNeighborsClassifier(n_neighbors=10)),
    ('svm_clf', SVC(C=0.7, kernel='rbf',probability=True)),
    ('dt_clf', DecisionTreeClassifier(max_depth=20)),
    ('mlp_clf',MLPClassifier())
], voting='soft')

start = time.time()

voting_clf2.fit(x_train, y_train)
print('The accuacy of training with soft voting is %f',voting_clf2.score(x_test, y_test))

voting_clf2.fit(c1,a1)
print('The accuacy of test with soft voting is %f',voting_clf2.score(c1,a1))

end = time.time()
print (end-start)

#########################################################################################3
f = open('DT1.pkl', "wb+")
pkl.dump(clf, f)
f.close()
