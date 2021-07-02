import math
import numpy
from sklearn.cluster import KMeans 

def sequence2Vector(sequence,c):
    '''
    クラスタ遷移の系列を、各遷移がどれくらい現れているかの頻度を成分とするベクトルに変換する関数
    クラスタリングの結果、クラスタがc個できるとすると遷移は全部でc**2個考えられる
    つまり、一つの系列は次元がk**2のベクトルに変換されることになる

    state_i -> state_jの遷移は、変換されるvectorにおけるi*c + j番目の成分になる。
    例えば
        c = 5
        state2 -> state 4への遷移の頻度は、ベクトルの14番目の成分になる
    '''

    vector = [0] * (c**2)
    length = len(sequence)
    for i in range(length-1):
        vector[c*sequence[i] + sequence[i+1]] += 1

    for i in range(c**2):
        vector[i] /= (length-1) 
    return vector

def anomaly_Detection(databese,c,clusters,eps):
    '''
    databaseに保存されている系列の集合を全てsequence2Vectorでベクトルに変関してから、point-baseの異常検知を行う
    クラスタリングはk-meansなどを用いて行い、ベクトル間の距離はユークリッド距離を用いることにする
    実験結果から、異常度の高い遷移には重みをつけて更新していくことも考えられると思う

    k-meansによる大きいクラスタを正常とみなしてそこから外れる系列を異常と判定する
    '''

    size = len(database)
    vectors = []
    cluster_size = [0] * clusters

    for i in range(size):
        vectors.append( sequence2Vector(database[i],c) )
    
    kmeans = KMeans(n_clusters=clusters,random_state=0).fit(vectors)
    for i in  range(size):
        cluster_size[kmeans.labels_[i]] += 1
    print(cluster_size)
    print(kmeans.labels_)

c = 5
database = [[1,2,3,4,0],[1,1,2,4],[0,4,4,2,1],[2,2,3,1,4],[1,1],[1,1,1,1,1,1,1,1,1,1]]
anomaly_Detection(database,c,2,0.0)