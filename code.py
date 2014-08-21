import pandas as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from AMSMetric import AMS

#resource.setrlimit(resource.RLIMIT_AS, (10000 * 1048576L, -1L))
print('reading files')
paths = ['training.csv', 'test.csv']
t = p.read_csv(paths[0])
#t2 = p.read_csv(paths[1])

print(len(t.count()))
X = np.asarray(t[t.columns[1 : len(t.count())-2]])


#test = tfidf.transform(t2['tweet']).toarray()
y = np.array(t['Label'])
le = preprocessing.LabelEncoder()
le.fit(y)
y=le.transform(y)
print(y[0:100])

w = np.array(t['Weight'])

clf = linear_model.SGDClassifier()
#clf.fit(X,y,sample_weight=w)
#print(clf.score(X,y))

def get_score(y,y_pred,weight):
    s=0
    b=0
    for index in range(len(y)):
        if(y[index]=='s'):
            if(y_pred[index] == 's'):
                s+=1
            else :
                b+=1
    return AMS(s,b)
    
def cross_validate(clf,X,y,w,n):
    kf = cross_validation.KFold(len(y), n_folds=n)
    def run(train_index,test_index):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        w_train = y[train_index]
        w_test = w[test_index]
        
        clf.fit(X_train,y_train,sample_weight=w_train)
        y_pred=le.inverse_transform(clf.predict(X_test))
        print(y_pred[0:100])
        return get_score(y_test,y_pred,w_test)
    return [run(train_index,test_index) for train_index, test_index in kf]
            

mse_scorer = make_scorer(get_score)
scores = cross_validate(clf,X,y,w,10)
print(scores)