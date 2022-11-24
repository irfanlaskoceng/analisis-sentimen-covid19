from flask import Flask, render_template, redirect, url_for, jsonify, request, session

# import time
import re,string
from nltk.util import pr
import pandas as pd
#from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from googletrans import Translator
#from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import pymysql


app = Flask(__name__)
app.secret_key = b'\xd6<\x90}\xb4+0\xc5c0\x07/\x1az\xf6V'

def caseFolding(text):
    #lowercase
    lower=text.lower()
    #hapus hastag/mention/retweet(RT)
    HastagRT=re.sub(r"#(\w+)|@(\w+)|(\brt\b)"," ", lower)
    #hapus URL
    pola_url = r'http\S+'
    CleanURL=re.sub(pola_url," ", HastagRT)
    #hapus angka
    hps_angka=re.sub(r"\d+", "", CleanURL)
    #hapus simbol
    hps_simbol = hps_angka.translate(str.maketrans("","",string.punctuation))
    #hapus singleChar, ex: q
    sChar = re.sub(r"\b[a-zA-Z]\b", "", hps_simbol)
    #hapus multiWhitespace++, ex: a   haa
    text = re.sub('\s+',' ',sChar)
    #hasil akhir casefolding
    hasil=text
    return hasil

def normal_term():
    normalisasi_word = pd.read_excel("_normalisasi.xlsx")
    normalisasi_dict = {}
    for index, row in normalisasi_word.iterrows():
        if row[0] not in normalisasi_dict:
            normalisasi_dict[row[0]] = row[1]
    return normalisasi_dict
def normalisasi(document):
    token_list = document.split()
    normalisasi_dict = normal_term() #import excel
    for term in range(len(token_list)):
        if token_list[term] in normalisasi_dict:
            token_list[term]=normalisasi_dict[token_list[term]]
            
    kalimat = " ".join(token_list).strip(" ")
    return kalimat

def tokenize(kalimat):
    return word_tokenize(kalimat)

def delstopwordID(teks):
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend(["dll", "dst","am","nya","nyaa"])
    list_stopwords = set(list_stopwords)
    return [kata for kata in teks if kata not in list_stopwords]
def delstopwordEN(teks):
    stopwordEN = set(stopwords.words('english'))
    return [kata for kata in teks if kata not in stopwordEN]

def stemming(kalimat):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    term_dict={}
    for kata in kalimat:
        for term in kalimat:
            if term not in term_dict:
                term_dict[term]=" "
    for term in term_dict:
        term_dict[term] = stemmer.stem(term)
    kalimat=[term_dict[term] for term in kalimat]
    listToStr = ' '.join([str(i) for i in kalimat])
    return listToStr

@app.route("/")
def index():
    try:
        connection = pymysql.connect(host='localhost', user='root', password='', db='dbfga')
        cursor = connection.cursor()

        sql = "SELECT * FROM `dataset`"
        cursor.execute(sql)
        dataset = cursor.fetchall()

        sql = "SELECT * FROM `final_prepocessing`"
        cursor.execute(sql)
        data_final_prepocessing = cursor.fetchall()

        if len(dataset)==0:
            print('insertttttttttttttttttttttt tweet awal')
            namaFile="Tweet_dgnSentimen_tes.csv"
            tweet_awal = pd.read_csv(namaFile)
            sql = "INSERT INTO `dataset` (`tweet`, `sentiment`) VALUES (%s, %s)"
            for row in tweet_awal.values.tolist():
                cursor.execute(sql, (row[0],row[1]))
            connection.commit()

    except Error as e:
        print(e)

    finally:
        connection.close()

    df_dataset = pd.DataFrame(list(dataset), columns =['ID','TWEET', 'SENTIMENT'])
    print('CaseFolding ...........................................')
    df_dataset['CaseFolding'] = df_dataset['TWEET'].apply(caseFolding)
    dts_casefolding =df_dataset[['ID','CaseFolding','SENTIMENT']].values.tolist()

    print('Normalized ...........................................')
    df_dataset['Normalized'] = df_dataset['CaseFolding'].apply(normalisasi)
    dts_normalized =df_dataset[['ID','Normalized','SENTIMENT']].values.tolist()

    print('Tokenisasi ...........................................')
    df_dataset['Tokenisasi'] = df_dataset['Normalized'].apply(tokenize)
    dts_tokenisasi =df_dataset[['ID','Tokenisasi','SENTIMENT']].values.tolist()

    index_names = df_dataset[df_dataset['SENTIMENT'] == 'Netral' ].index
    df_dataset.drop(index_names, inplace = True)
    df_dataset.reset_index()
    dts_hps_netral =df_dataset[['ID','Tokenisasi','SENTIMENT']].values.tolist()
    
    print('Stopword ...........................................')
    df_dataset['StopwordID'] = df_dataset['Tokenisasi'].apply(delstopwordID)
    df_dataset['StopwordEN'] = df_dataset['StopwordID'].apply(delstopwordEN)
    dts_stopword =df_dataset[['ID','StopwordEN','SENTIMENT']].values.tolist()

    if len(data_final_prepocessing)==0:
        try:
            print('insertttttttttttttttttttttt final prepocessing')
            connection = pymysql.connect(host='localhost', user='root', password='', db='dbfga')
            cursor = connection.cursor()
        
            df_dataset['Stemmed'] = df_dataset['StopwordEN'].apply(stemming)
            dts_stemmed =df_dataset[['Stemmed','SENTIMENT']].values.tolist()            
            sql = "INSERT INTO `final_prepocessing` (`tweet_FP`, `sentiment_FP`) VALUES (%s, %s)"
            for row in dts_stemmed:
                cursor.execute(sql, (row[0],row[1]))
            connection.commit()
            sql = "SELECT * FROM `final_prepocessing`"
            cursor.execute(sql)
            data_final_prepocessing = cursor.fetchall()
        except Error as e:
            print(e)
        finally:
            connection.close()
    # else:
    #     try:
    #         print('delllllllllllllllllllllll')
    #         connection = pymysql.connect(host='localhost', user='root', password='', db='dbfga')
    #         cursor = connection.cursor()
    #         sql = "DELETE FROM `final_prepocessing`"
    #         cursor.execute(sql)
    #         connection.commit()
    #     except Error as e:
    #         print(e)
    #     finally:
    #         connection.close()

    return render_template("home.html", dataset=dataset, dts_casefolding=dts_casefolding, dts_normalized=dts_normalized,
    dts_tokenisasi=dts_tokenisasi, dts_hps_netral=dts_hps_netral, dts_stopword=dts_stopword, data_final_prepocessing=data_final_prepocessing)

@app.route("/klasifikasiKNN")
def klasifikasiKNN():
    tmp_training = request.args.get('d_training', 0, type=int)
    tmp_testing = request.args.get('d_testing', 0, type=int)
    tmp_testing=tmp_testing/100

    try:
        connection = pymysql.connect(host='localhost', user='root', password='', db='dbfga')
        cursor = connection.cursor()

        sql = "SELECT * FROM `final_prepocessing`"
        cursor.execute(sql)
        data_final_prepocessing = cursor.fetchall()

    except Error as e:
        print(e)

    finally:
        connection.close()

    df_final_pre = pd.DataFrame(list(data_final_prepocessing), columns =['TWEET', 'SENTIMENT'])
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(df_final_pre['TWEET'], df_final_pre['SENTIMENT'], test_size = tmp_testing, random_state = 0)
    df_train = pd.DataFrame()
    df_train['TWEET'] = train_X
    df_train['SENTIMENT'] = train_Y

    df_test = pd.DataFrame()
    df_test['TWEET'] = test_X
    df_test['SENTIMENT'] = test_Y
    print('bagi data training dan testing selesai ...........................................')
    
    # print(df_final_pre)

    print('TF-IDF ...........................................')
  
    tfidf_vect = TfidfVectorizer(max_features = 5000)
    tfidf_vect.fit(df_final_pre['TWEET'])
    train_X_tfidf = tfidf_vect.transform(df_train['TWEET'])
    test_X_tfidf  = tfidf_vect.transform(df_test['TWEET'])

    fitur = tfidf_vect.get_feature_names()
    cek_train_X_tfidf_lebih_detail = pd.DataFrame(train_X_tfidf.toarray(), columns = fitur)
    cek_test_X_tfidf_lebih_detail = pd.DataFrame(test_X_tfidf.toarray(), columns = fitur)
    print('TF-IDF selesai...........................................')
    
    print('klasifikasi knn ...........................................')
    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(cek_train_X_tfidf_lebih_detail,train_Y)
    prediction = model.predict(cek_test_X_tfidf_lebih_detail)

    join_test_tweet_and_prediction = pd.DataFrame()
    join_test_tweet_and_prediction['TWEET'] = test_X
    join_test_tweet_and_prediction['SENTIMENT'] = test_Y
    join_test_tweet_and_prediction['PREDICTION'] = prediction
    result_accuracy = accuracy_score(prediction, test_Y)*100
    result_accuracy = round(result_accuracy,1)

    print('inisiasi tampilan hasil kasifikasi')
    tmp=''''''
    # menampilkan data traning
    tmp+=''' <div class="row mt-3">'''
    tmp+='''            <div  class="col-sm-6">
                    <h5>Data Training</h5>
                    <div class="card bgCARD">
                        <div class="table-responsive " style="height: 500px; overflow-y: auto;">
                            <table class="table  table-sm table-bordered">
                                
                                <thead>
                                    <tr>
                                        <th>Tweet</th>
                                        <th>Sentiment</th>
                                    </tr>
                                </thead>
                                <tbody>'''
    for row in df_train.values.tolist():
        tmp+='''                    <tr>'''
        for col in row:
            tmp+='''                    <td class="text-justify">'''+str(col)+'''</td>'''
        tmp+='''                    </tr> '''
    tmp+='''                    </tbody>    
                            </table>
                        </div>
                    </div>
                </div>'''
    
    # menampilkan data testing
    tmp+='''            <div  class="col-sm-6">
                    <h5>Data Testing</h5>
                    <div class="card bgCARD">
                        <div class="table-responsive " style="height: 500px; overflow-y: auto;">
                            <table class="table  table-sm table-bordered">
                                
                                <thead>
                                    <tr>
                                        <th>Tweet</th>
                                        <th>Sentiment</th>
                                    </tr>
                                </thead>
                                <tbody>'''
    for row in df_test.values.tolist():
        tmp+='''                    <tr>'''
        for col in row:
            tmp+='''                    <td class="text-justify">'''+str(col)+'''</td>'''
        tmp+='''                    </tr> '''
    tmp+='''                    </tbody>    
                            </table>
                        </div>
                    </div>
                </div>'''

    
    tmp+='''</div>'''



    
    
    # # menampilkan tf-idf traning
    # tmp+=''' <div class="row mt-3">'''
    # tmp+='''            <div  class="col-sm-6">
    #                 <h5>TF-IDF Data Training</h5>
    #                 <div class="card bgCARD">
    #                     <div class="table-responsive " style="height: 500px; overflow-y: auto;">
    #                         <table class="table  table-sm table-bordered">
                                
    #                             <thead>'''
    # tmp+='''                        <tr>'''
    # for f_i in fitur:
    #     tmp+='''                        <th class="text-justify">'''+str(f_i)+'''</th>'''
    # tmp+='''                        </tr>
    #                             </thead>
    #                             <tbody>'''
    # for row in cek_train_X_tfidf_lebih_detail.values.tolist():
    #     tmp+='''                    <tr>'''
    #     for col in row:
    #         tmp+='''                    <td class="text-justify">'''+str(round(col,4))+'''</td>'''
    #     tmp+='''                    </tr> '''
    # tmp+='''                    </tbody>    
    #                         </table>
    #                     </div>
    #                 </div>
    #             </div>'''

    # # menampilkan tf-idf testing
    # tmp+='''            <div  class="col-sm-6">
    #                 <h5>TF-IDF Data Testing</h5>
    #                 <div class="card bgCARD">
    #                     <div class="table-responsive " style="height: 500px; overflow-y: auto;">
    #                         <table class="table  table-sm table-bordered">
                                
    #                             <thead>'''
    # tmp+='''                        <tr>'''
    # for f_i in fitur:
    #     tmp+='''                        <th class="text-justify">'''+str(f_i)+'''</th>'''
    # tmp+='''                        </tr>
    #                             </thead>
    #                             <tbody>'''
    # for row in cek_test_X_tfidf_lebih_detail.values.tolist():
    #     tmp+='''                    <tr>'''
    #     for col in row:
    #         tmp+='''                    <td class="text-justify">'''+str(round(col,4))+'''</td>'''
    #     tmp+='''                    </tr> '''
    # tmp+='''                    </tbody>    
    #                         </table>
    #                     </div>
    #                 </div>
    #             </div>'''
    # tmp+='''</div>'''


    #menampilkan prediksi data testing
    tmp+=''' <div class="row mt-3">'''
    tmp+='''
                <div  class="col-sm-6">
                    <h5>Prediction</h5>
                    <div class="card bgCARD">
                        <div class="table-responsive " style="height: 500px; overflow-y: auto;">
                            <table class="table  table-sm table-bordered">
                                
                                <thead>
                                    <tr>
                                        <th>Tweet</th>
                                        <th>Sentiment</th>
                                        <th>Prediction</th>
                                    </tr>
                                </thead>
                                <tbody>'''
    for row in join_test_tweet_and_prediction.values.tolist():
        tmp+='''                    <tr>'''
        for col in row:
            tmp+='''                    <td class="text-justify">'''+str(col)+'''</td>'''
        tmp+='''                    </tr> '''
    tmp+='''                    </tbody>    
                            </table>
                        </div>
                    </div>
                </div>'''
    
    # menampilkan performa
    tmp+='''
                <div  class="col-sm-6">
                    <h5>performance</h5>
                    <div class="card bgCARD  pl-2 pt-2">
                        <p>Akurasi : '''+str(result_accuracy)+'''%</p>
                        
                    </div>
                </div>'''
    tmp+='''</div>'''
    


    print((df_train), len(train_Y))
    print(join_test_tweet_and_prediction)
    print(fitur)
    return jsonify(result=tmp)
# @app.route("/re_prepocessing")
