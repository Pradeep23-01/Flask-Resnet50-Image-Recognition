import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


st.title('Find a Suitable Machine learning model for a dataset')

st.write("""
	Explore different datasets and find out which classifier model to use for better accuracy. """)

dataset_name=st.sidebar.selectbox("Select Dataset",("Iris","Breast cancer","Wine dataset"))

classifier_name=st.sidebar.selectbox('Select Model',('KNN','SVM','Random forest'))

def get_dataset(dataset_name):

	if dataset_name=='Iris':
		data=datasets.load_iris()

	elif dataset_name=='Breast cancer':
		data=datasets.load_breast_cancer()

	else:
		data=datasets.load_wine()

	X=data.data
	Y=data.target

	return X,Y

X,Y=get_dataset(dataset_name)

st.write('Shape of the dataset ',X.shape)
st.write('Number of classes ',len(np.unique(Y)))

def add_parameter_ui(classifier_name):
	params=dict()
	if classifier_name=='KNN':
		K=st.sidebar.slider('K',1,15)
		params["K"]=K
	elif classifier_name=='SVM':
		C=st.sidebar.slider("C",0.01,1.0)
		params["C"]=C
	else:
		max_depth=st.sidebar.slider("max_depth",2,15)
		n_estimators=st.sidebar.slider("n_estimators",2,15)
		params["max_depth"]=max_depth
		params["n_estimators"]=n_estimators

	return params
 
params=add_parameter_ui(classifier_name)


def get_classifier(params,classifier_name):

	if classifier_name=='KNN':
		clf=KNeighborsClassifier(n_neighbors=params['K'])
	elif classifier_name=='SVM':
		clf=SVC(C=params['C'])
	else:
		clf=RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=1234)

	return clf

clf=get_classifier(params,classifier_name)

#Classification
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1234)
clf.fit(X_train,Y_train)

y_pred=clf.predict(X_test)

acc=accuracy_score(y_pred,Y_test)

st.write(f'Classifier={classifier_name}')
st.write(f'accuracy={acc}')

pca=PCA(2)

X_projected=pca.fit_transform(X)

x1=X_projected[:,0]
x2=X_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=Y,alpha=0.8,cmap='viridis')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')

plt.colorbar()
#plt.show()

st.pyplot(fig)


