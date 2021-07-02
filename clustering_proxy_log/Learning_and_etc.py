import log_attached_blacklist as moto#ログからのDF作成に関するメソッドを定義
import pandas as pd
from sklearn.externals import joblib#オブジェクトの保存・読み込みに利用
from os.path import splitext#urlのpath部分から最後のドット以降を抽出するのに使用
from os.path import exists#ファイルの有無を確認するのに仕様
import numpy as np
import math
from time import time#時間計測に利用

from sklearn import svm#linear-svm,svm
from sklearn.cluster import KMeans#K平均法
from sklearn.mixture import GMM#Gaussian Mixture Model
from sklearn.cluster import DBSCAN#DBSCAN
from sklearn.model_selection import train_test_split # ver1.8以降はモジュールの場所が変更されている#データセットをtrainとtestに分割する


def load_DF(dfdir):
    """
        DFsフォルダ以下にあるDataFrameを読みこむメソッド(自分がわかりやすいために利用)
    """
    DF = joblib.load("./DFs/"+dfdir);
    return DF

def Make_train_and_test_DF(in_dfdir,check=True):
    """
        Input_DFsフォルダ以下のDataFrameを読んで、TestとTrainigに分ける
        checkがTrueならば既に分割済みであれば何もしない,Falseなら新たに分割を行う
    """
    if(not(check and exists("./DFs/Train_DFs/train_"+in_dfdir))):#checkがTrueで既に分割ファイルが有るときのみ何もしない
        DF = load_DF("Input_DFs/"+in_dfdir)
        train_DF, test_DF= train_test_split(DF, test_size=0.1, random_state=4)#データセットを訓練用とテスト用に分割
        joblib.dump(train_DF,"./DFs/Train_DFs/train_"+in_dfdir)
        joblib.dump(test_DF,"./DFs/Test_DFs/test_"+in_dfdir)

def trainKMeans(dfdir,cmin=2,cmax=7,mode=False):
    """
        dfdir:Test_DFs/train_以降の名前を指定(指定通りに作成していればInputで始まるDF)
        cmin,cmax:それぞれいくつに分割するかの下限，上限
        mode:Trueのときは各実行時に時間を計測して表示(print)
    """
    train_DF = load_DF("Train_DFs/train_"+dfdir)
    if(mode):print("KMeans Train train_"+dfdir)
    for j in range(cmin,cmax):
        km=KMeans(n_clusters=j)
        if(mode):print("n_clusters="+str(j));t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
        km=km.fit(train_DF)
        if(mode):t1=time()-t0;print("Training in {} seconds".format(round(t1,3,)));print()#predict時間の表示
        joblib.dump(km,"./Classifiers/KMeans/km_"+str(j)+"_"+dfdir)
        
def testKMeans(dfdir,cmin=2,cmax=7,mode=False):
    """
        dfdir:Test_DFs/test_以降の名前を指定(指定通りに作成していればInputで始まるDF)
        cmin,cmax:それぞれいくつに分割するかの下限，上限
    """
    test_DF = load_DF("Test_DFs/test_"+dfdir)
    if(mode):print("KMeans Test train_"+dfdir)
    for j in range(cmin,cmax):
        km = joblib.load("./Classifiers/KMeans/km_"+str(j)+"_"+dfdir)
        if(mode):print("n_clusters="+str(j));t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
        result_km=km.predict(test_DF)
        if(mode):t1=time()-t0;print("Testing in {} seconds".format(round(t1,3,)));print()#predict時間の表示
        joblib.dump(pd.DataFrame(result_km,columns=["label"],index=test_DF.index),"./DFs/Results/KMeans/Result_km_"+str(j)+"_test_"+dfdir)#結果をDFにして保存
        print("n_clusters = %d" % j)#何クラスタに分類したかを表示
        temp=""#次に表示するものの空文字列を用意
        for i in range(j):
            temp+="%d:%d " %(i,len(np.where(result_km==i)[0]))#各クラスタに分類された数を計算して文字列に追加
        print(temp)
        centers = km.cluster_centers_
        print(centers)#各クラスタの各次元の中心点の値を表示
#        for line in test_DF.index:
#            dis_list=[]#各点と各クラスタの距離
#            for cluster_center in centers:
#                temp=0.0
#                for column in test_DF.columns:
#                    temp = temp + math.pow(float(test_DF.loc[line,column])-float(cluster_center[list(test_DF.columns).index(column)]),2.0)#この点とクラスタの中心との距離を計算
#                temp = math.sqrt(temp)
#                dis_list.append(temp)
            #print(dis_list);print(result_km[list(test_DF.index).index(line)])
        print(result_km)
        if(cmin+1==cmax):return result_km
        print()

def testKMeans_full(dfdir,cmin=2,cmax=7,mode=False):
    """
        dfdir:Input_DFs/以降の名前を指定(指定通りに作成していればInputで始まるDF)
        cmin,cmax:それぞれいくつに分割するかの下限，上限
    """
    test_DF = load_DF("Input_DFs/"+dfdir)
    if(mode):print("KMeans Test "+dfdir)
    for j in range(cmin,cmax):
        km = joblib.load("./Classifiers/KMeans/km_"+str(j)+"_"+dfdir)
        if(mode):print("n_clusters="+str(j));t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
        result_km=km.predict(test_DF)
        if(mode):t1=time()-t0;print("Testing in {} seconds".format(round(t1,3,)));print()#predict時間の表示
        joblib.dump(pd.DataFrame(result_km,columns=["label"],index=test_DF.index),"./DFs/Results/KMeans/Result_km_"+str(j)+"_full_"+dfdir)#結果をDFにして保存
        print("n_clusters = %d" % j)#何クラスタに分類したかを表示
        temp=""#次に表示するものの空文字列を用意
        for i in range(j):
            temp+="%d:%d " %(i,len(np.where(result_km==i)[0]))#各クラスタに分類された数を計算して文字列に追加
        print(temp)
        centers = km.cluster_centers_
        print(centers)#各クラスタの各次元の中心点の値を表示
#        for line in test_DF.index:
#            dis_list=[]#各点と各クラスタの距離
#            for cluster_center in centers:
#                temp=0.0
#                for column in test_DF.columns:
#                    temp = temp + math.pow(float(test_DF.loc[line,column])-float(cluster_center[list(test_DF.columns).index(column)]),2.0)#この点とクラスタの中心との距離を計算
#                temp = math.sqrt(temp)
#                dis_list.append(temp)
            #print(dis_list);print(result_km[list(test_DF.index).index(line)])
        print(result_km)
        if(cmin+1==cmax):return result_km
        print()

def train_and_testKMeans_full(dfdir,cmin=2,cmax=7,mode=False):
    """
        入力データ全てで学習してその結果を返す
        dfdir:Input_DFs/Input_以降の名前を指定(指定通りに作成していればDFで始まる)
        cmin,cmax:それぞれいくつに分割するかの下限，上限
        mode:Trueのときは各実行時に時間を計測して表示(print)
    """
    Input_DF = load_DF("Input_DFs/Input_"+dfdir)
    if(mode):print("KMeans Train Input_"+dfdir)
    for j in range(cmin,cmax):
        km=KMeans(n_clusters=j)
        if(mode):print("n_clusters="+str(j));t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
        km=km.fit(Input_DF)
        if(mode):t1=time()-t0;print("Training in {} seconds".format(round(t1,3,)));print()#predict時間の表示
        joblib.dump(km,"./Classifiers/KMeans/km_"+str(j)+"_"+dfdir)
        if(mode):print("n_clusters="+str(j));t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
        result_km=km.predict(Input_DF)
        if(mode):t1=time()-t0;print("Testing in {} seconds".format(round(t1,3,)));print()#predict時間の表示
        joblib.dump(pd.DataFrame(result_km,columns=["label"],index=test_DF.index),"./DFs/Results/KMeans/Result_km_"+str(j)+"_full_"+dfdir)#結果をDFにして保存
        
def trainGMM(dfdir,cmin=2,cmax=7,mode=False):
    """
        dfdir:Test_DFs/train_以降の名前を指定(指定通りに作成していればInputで始まるDF)
        cmin,cmax:それぞれいくつに分割するかの下限，上限
    """
    train_DF = load_DF("Train_DFs/train_"+dfdir)
    for j in range(cmin,cmax):
        gmm=GMM(n_components=j)
        if(mode):print("n_components="+str(j));t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
        gmm=gmm.fit(train_DF)
        if(mode):t1=time()-t0;print("Training in {} seconds".format(round(t1,3,)));print()#predict時間の表示
        joblib.dump(gmm,"./Classifiers/GMM/gmm_"+str(j)+"_"+dfdir)        
        
def testGMM(dfdir,cmin=2,cmax=7,mode=False):
    """
        dfdir:Test_DFs/test_以降の名前を指定(指定通りに作成していればInputで始まるDF)
        cmin,cmax:それぞれいくつに分割するかの下限，上限
    """
    test_DF = load_DF("Test_DFs/test_"+dfdir)
    for j in range(cmin,cmax):
        gmm = joblib.load("./Classifiers/GMM/gmm_"+str(j)+"_"+dfdir)
        if(mode):print("n_components="+str(j));t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
        result_gmm=gmm.predict(test_DF)
        if(mode):t1=time()-t0;print("Testing in {} seconds".format(round(t1,3,)));print()#predict時間の表示
        joblib.dump(pd.DataFrame(result_gmm,columns=["label"],index=test_DF.index),"./DFs/Results/GMM/Result_gmm_"+str(j)+"_test_"+dfdir)#結果をDFにして保存
        print("n_components = %d" % j)#何クラスタに分類したかを表示
        temp=""#次に表示するものの空文字列を用意
        for i in range(j):
            temp+="%d:%d " %(i,len(np.where(result_gmm==i)[0]))#各クラスタに分類された数を計算して文字列に追加
        print(temp)
        print(result_gmm)
        print()

def testGMM_full(dfdir,cmin=2,cmax=7,mode=False):
    """
        dfdir:Input_DFs/以降の名前を指定(指定通りに作成していればInputで始まるDF)
        cmin,cmax:それぞれいくつに分割するかの下限，上限
    """
    test_DF = load_DF("Input_DFs/"+dfdir)
    for j in range(cmin,cmax):
        gmm = joblib.load("./Classifiers/GMM/gmm_"+str(j)+"_"+dfdir)
        if(mode):print("n_components="+str(j));t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
        result_gmm=gmm.predict(test_DF)
        if(mode):t1=time()-t0;print("Testing in {} seconds".format(round(t1,3,)));print()#predict時間の表示
        joblib.dump(pd.DataFrame(result_gmm,columns=["label"],index=test_DF.index),"./DFs/Results/GMM/Result_gmm_"+str(j)+"_full_"+dfdir)#結果をDFにして保存
        print("n_components = %d" % j)#何クラスタに分類したかを表示
        temp=""#次に表示するものの空文字列を用意
        for i in range(j):
            temp+="%d:%d " %(i,len(np.where(result_gmm==i)[0]))#各クラスタに分類された数を計算して文字列に追加
        print(temp)
        print(result_gmm)
        print()

def trainDBSCAN(dfdir,mode=False):
    """
        dfdir:Test_DFs/train_以降の名前を指定(指定通りに作成していればInputで始まるDF)
        cmin,cmax:それぞれいくつに分割するかの下限，上限
    """
    train_DF = load_DF("Train_DFs/train_"+dfdir)
    dbscan=DBSCAN()
    print(dbscan)
    if(mode):t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
    dbscan=dbscan.fit(train_DF)
    if(mode):t1=time()-t0;print("Training in {} seconds".format(round(t1,3,)));print()#predict時間の表示
    joblib.dump(dbscan,"./Classifiers/DBSCAN/dbscan_"+dfdir)        
        
def testDBSCAN(dfdir,mode=False):
    """
        dfdir:Test_DFs/test_以降の名前を指定(指定通りに作成していればInputで始まるDF)
        cmin,cmax:それぞれいくつに分割するかの下限，上限
    """
    test_DF = load_DF("Test_DFs/test_"+dfdir)
    dbscan = joblib.load("./Classifiers/DBSCAN/dbscan_"+dfdir)
    if(mode):t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
    result_dbscan=dbscan.fit_predict(test_DF)
    if(mode):t1=time()-t0;print("Testing in {} seconds".format(round(t1,3,)));print()#predict時間の表示
    joblib.dump(pd.DataFrame(result_dbscan,columns=["label"],index=test_DF.index),"./DFs/Results/DBSCAN/Result_dbscan_test_"+dfdir)#結果をDFにして保存
    temp=""#次に表示するものの空文字列を用意
    print(temp)
    print(result_dbscan)
    print()

def testDBSCAN_full(dfdir,mode=False):
    """
        dfdir:Input_DFs/以降の名前を指定(指定通りに作成していればInputで始まるDF)
        cmin,cmax:それぞれいくつに分割するかの下限，上限
    """
    test_DF = load_DF("Input_DFs/"+dfdir)
    dbscan = joblib.load("./Classifiers/DBSCAN/dbscan_"+dfdir)
    if(mode):t0 = time();print("Start in {} ".format(round(t0,3,)))#訓練開始時刻の表示
    result_dbscan=dbscan.fit_predict(test_DF)
    if(mode):t1=time()-t0;print("Testing in {} seconds".format(round(t1,3,)));print()#predict時間の表示
    joblib.dump(pd.DataFrame(result_dbscan,columns=["label"],index=test_DF.index),"./DFs/Results/DBSCAN/Result_dbscan_full_"+dfdir)#結果をDFにして保存
    temp=""#次に表示するものの空文字列を用意
    print(temp)
    print(result_dbscan)
    print()
