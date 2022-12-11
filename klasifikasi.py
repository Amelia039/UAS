import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib
from joblib import load
from sklearn.preprocessing import LabelEncoder
from PIL import Image


st.title("APLIKASI WEB KLASIFIKASI")
st.write("""
# Klasifikasi Breast Cancer Wisconsin
Web Apps for Breast Cancer Wisconsin (Diagnostic) Data Set Classification
         
         """
)
st.write("""
#Amelia Nur Septiyasari - 200411100039 - Penambangan Data IF5A    
         """
)

img= Image.open('env\image.jpg')
st.image(img,use_column_width=False)

deskripsi,import_data, tahap_preprocessing, tahap_modeling, implementation = st.tabs(["Deskripsi Data","Import Data", "Prepocessing", "Modeling", "Implementation"])
df=pd.read_csv("https://raw.githubusercontent.com/Amelia039/my-dataset/main/wdbc.csv")
with deskripsi:
     st.write("""- Kanker payudara (Carcinoma mamae) 
              dalam bahasa inggrisnya disebut breast cancer merupakan kanker pada
jaringan payudara. Saat ini, kanker payudara merupakan
penyebab kematian kedua akibat kanker pada wanita, setelah
kanker leher rahim, dan merupakan kanker yang paling banyak
ditemui diantara wanita.
Ada beberapa tahapan pemeriksaan lebih lanjut dalam
mendeteksi penyakit kanker payudara apakah termasuk dalam
kategori jinak atau ganas""")
     st.write("""Dataset Wisconsin Diagnostic merupakan dataset populer dan secara luas digunakan oleh peneliti data mining untuk mendiagnosis penyakit kanker payudara""")
     st.write("""# Tipe Data""")
     result = df.dtypes
     result
     st.write("""# sumber data link dataset
              https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
              """)
     st.write("""# Fitur dalam dataset""")
     "id"
     st.caption("Merupakan nomor identitas pasien")
     "diagnosis"
     st.caption("Merupakan label diagnosis pasien yang terkena penyakit kanker payudara(M = ganas, B = jinak)")
     "radius"
     st.caption("Merupakan rata-rata jarak dari pusat ke titik-titik pada keliling payudara")
     st.latex(r'''
              r = \frac{d}{2}

              ''')
     "texture"
     st.caption("Merupakan standar deviasi nilai skala abu-abu (standard deviation of gray-scale values)")
     st.latex(r'''s=\sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \overline{x})^2}''')
     "perimeter"
     st.caption("keliling payudara ")
     st.latex(r''' P = 2\pi r''')
     "area"
     st.caption("luas daerah payudara")
     st.latex(r'''A = \pi r^2''')
     " smoothness"
     st.caption('kehalusan (variasi lokal dalam panjang radius)')
     "compactness"
     st.caption("kekompakan")
     st.latex(r'''\frac{keliling^2}{luas - 1,0 }''')
     "concavity"
     st.caption("kekompakan")
     st.latex(r'''\frac{keliling^2}{luas - 1,0 }''')
     " concave_points"	
     st.caption("kekompakan")
     st.latex(r'''\frac{keliling^2}{luas - 1,0 }''')
     "symmetry"
     st.caption("kekompakan")
     st.latex(r'''\frac{keliling^2}{luas - 1,0 }''')
     "fractal dimension"
     st.caption("kekompakan")
     st.latex(r'''\frac{keliling^2}{luas - 1,0 }''')
     
with import_data:
   
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category

    st.write("""# Tahapan untuk mengimpor Dataset""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Dataset= ", uploaded_file.name)
        st.dataframe(df)
        
with tahap_preprocessing:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category

    st.write("""# Tahapan untuk melakukan Preprocessing""")
   
    
    df = df.drop(columns=["id","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"])
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.diagnosis).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[1]],
        '0' : [dumies[0]]
    })

    st.write(labels)
    
with tahap_modeling:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category
    X_train, X_test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    st.write("""# Tahap Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    naivebayes = GaussianNB()
    naivebayes = naivebayes.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = naivebayes.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    naivebayes.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    joblib.dump(naivebayes, 'naivebayes.joblib')
    

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    
    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))
    joblib.dump(knn, 'knn.joblib')
    
    # DT

    destree = DecisionTreeClassifier()
    destree.fit(X_train, y_train)
    # prediction
    destree.score(X_test, y_test)
    y_pred = destree.predict(X_test)
    #Accuracy
    accuracy = round(100 * accuracy_score(y_test,y_pred))
    joblib.dump(destree, 'destree.joblib')
    

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(accuracy))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,accuracy],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

with implementation:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category

    st.write("# Implementation")
    radius_mean	= st.number_input('Masukkan radius mean')
    texture_mean = st.number_input('Masukkan texture mean')
    perimeter_mean = st.number_input('Masukkan perimeter mean')
    area_mean = st.number_input('Masukkan area mean')
    smoothness_mean	= st.number_input('Masukkan smoothness mean')
    compactness_mean = st.number_input('Masukkan compactness mean')
    concavity_mean = st.number_input('Masukkan concavity mean')
    concave_points_mean	= st.number_input('Masukkan concave points mean')
    symmetry_mean = st.number_input('Masukkan symetry mean')
    fractal_dimension_mean= st.number_input('Masukkan fractal mean')
    
    cek_hasil = st.button("Cek Prediksi")

    knn = joblib.load("knn.joblib")
    destree = joblib.load("destree.joblib")
    naivebayes = joblib.load("naivebayes.joblib")
    
    #============================ Mengambil akurasi tertinggi ===========================
    if akurasi > skor_akurasi and accuracy:
        use_model=naivebayes
        metode = "Naive bayer"
    elif skor_akurasi > akurasi and accuracy:
        use_model=knn
        metode = "KNN"
    elif accuracy > skor_akurasi and akurasi:
        use_model=destree
        metode = "Decission Tree"
    #============================ Normalisasi inputan =============================
    inputan = np.array([radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean])
    df_min = X.min()
    df_max = X.max()
    input_norm = ((inputan - df_min) / (df_max - df_min))
    input_norm = np.array(input_norm).reshape(1, -1)

   
    # inputan
    # inputan_norm
    if cek_hasil:
        input_pred = use_model.predict(input_norm)[0]

        st.subheader('Hasil Prediksi')
        if input_pred == 1:
            st.error('Anda didiagnosis terkena kanker payudara Jinak')
        else:
            st.success('Anda didiagnosis terkena kanker payudara Ganas')