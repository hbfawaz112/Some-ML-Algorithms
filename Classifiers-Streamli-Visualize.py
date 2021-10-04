import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from mpl_toolkits.mplot3d import axes3d
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def return_data(dataset):
    if dataset == 'Wine':
        data = load_wine()
    elif dataset == 'Iris':
        data = load_iris()
    else:
        data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names , index=None)
    df['Type'] = data.target
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=1, test_size=0.2)
    return X_train, X_test, y_train, y_test,df,data.target_names


def getClassifier(classifier):
    if classifier == 'SVM':
        c = st.sidebar.slider(label='Chose value of C' , min_value=0.0001, max_value=10.0)
        model = SVC(C=c)
    elif classifier == 'KNN':
        neighbors = st.sidebar.slider(label='Chose Number of Neighbors',min_value=1,max_value=20)
        model = KNeighborsClassifier(n_neighbors = neighbors)
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 10)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        model = RandomForestClassifier(max_depth = max_depth , n_estimators= n_estimators,random_state= 1)
    return model


def getPCA(df):
    pca = PCA(n_components=3)
    result = pca.fit_transform(df.loc[:,df.columns != 'Type'])
    df['pca-1'] = result[:, 0]
    df['pca-2'] = result[:, 1]
    df['pca-3'] = result[:, 2]
    return df

# Title
st.title("Classifiers in Action")

# Description
st.text("Chose a Dataset and a Classifier in the sidebar. Input your values and get a prediction")

#sidebar
sideBar = st.sidebar
dataset = sideBar.selectbox('Which Dataset do you want to use?',('Wine' , 'Breast Cancer' , 'Iris'))
classifier = sideBar.selectbox('Which Classifier do you want to use?',('SVM' , 'KNN' , 'Random Forest'))
# Get Data
X_train, X_test, y_train, y_test, df , classes= return_data(dataset)
st.dataframe(df.sample(n = 5 , random_state = 1))
st.subheader("Classes")
for idx, value in enumerate(classes):
    st.text('{}: {}'.format(idx , value))

# 2-D PCA
df = getPCA(df)
fig = plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-1", y="pca-2",
    hue="Type",
    palette=sns.color_palette("hls", len(classes)),
    data=df,
    legend="full"
)
plt.xlabel('PCA One')
plt.ylabel('PCA Two')
plt.title("2-D PCA Visualization")
st.pyplot(fig)

#3-D PCA
fig2 = plt.figure(figsize=(16,10)).gca(projection='3d')
fig2.scatter(
    xs=df["pca-1"],
    ys=df["pca-2"],
    zs=df["pca-3"],
    c=df["Type"],
)
fig2.set_xlabel('pca-one')
fig2.set_ylabel('pca-two')
fig2.set_zlabel('pca-three')
st.pyplot(fig2.get_figure())

# Train Model
model = getClassifier(classifier)
model.fit(X_train, y_train)
test_score = round(model.score(X_test, y_test), 2)
train_score = round(model.score(X_train, y_train), 2)

st.subheader('Train Score: {}'.format(train_score))
st.subheader('Test Score: {}'.format(test_score))