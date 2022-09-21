
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('heart disease classification dataset.csv')
print(data.head())

data.dropna(inplace = True)
data.reset_index(drop= True,inplace = True)
print(data.head())

data['sex'] = data['sex'].map({'male': 1, 'female': 0})

X = data.drop(['Unnamed: 0', 'target'], axis = 1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size= 0.80, test_size=0.20, shuffle= True)

class Classifier:

    def support_vect():
        svm = SVC()
        svm.fit(X_train, y_train)
        p = svm.predict(X_test)
        print(f"Accuracy of SVM: {accuracy_score(y_test, p)*100}")

    def gausian():
        gauss = GaussianNB()
        gauss.fit(X_train, y_train)
        p = gauss.predict(X_test)
        print(f"Accuracy of Naive Bayes: {accuracy_score(y_test, p)*100}")

    def des_tree():
        dst = DecisionTreeClassifier()
        dst.fit(X_train,y_train)
        p = dst.predict(X_test)
        print(f"Accuracy of Decision Tree: {accuracy_score(y_test, p)*100}")

    def knearest(): 
        knn =  KNeighborsClassifier(n_neighbors=30)
        knn.fit(X_train,y_train)
        p = knn.predict(X_test)
        print(f"Accuracy of Knearest Neighbor: {accuracy_score(y_test, p)*100}")

while True:
    n = int(input("\nChoices for Classifier:\n \t1) Support Vector Machine\n\t2) Naive Bayes\n\t3) Descision Tree\n\t4) Knearest Neighbors\n\t:= "))
    if n == 'q':
        break 

    if n == 1:
        Classifier.support_vect()
    elif n == 2:
        Classifier.gausian()
    elif n == 3:
        Classifier.des_tree()
    elif n == 4:
        Classifier.knearest()
    else:
        print("\tinvalid choice")
    
        




