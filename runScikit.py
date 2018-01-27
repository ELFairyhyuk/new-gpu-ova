from sklearn.datasets import load_svmlight_files
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn
X_train, y_train, X_test, y_test = load_svmlight_files(("/path/to/train_dataset.txt", "/path/to/test_dataset.txt”))
X_train=X_train.toarray()
X_test=X_test.toarray()
nb_classif=OneVsRestClassifier(estimator=GaussianNB() #nb_classif=GaussianNB()
nb_classif.fit(X_train,y_train)
y_pred=Classifiers.predict(X_test)
accuracy_score(y_test, y_pred)
