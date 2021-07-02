import numpy as np
from scipy.stats import chi2

def sequence2Vector(sequence,c):
    '''
    系列を方向ベクトルに変換する関数

    STEP1
    state_i -> state_jの遷移はvector[c*i + j]として保存する

    STEP2:
    長さを１に正規化する
    '''
    
    length = len(sequence)
    vector = [0] * (c**2)
    for i in range(length-1):
        vector[c*sequence[i] + sequence[i+1]] += 1

    norm = np.linalg.norm(vector)
    for i in range(c**2):
        vector[i] /= norm

    return vector

'''
def learn_Kappa(directional_database,mean):
    a(x) = 1 - np.dot(mean,x)
    a(x)の分布がもっとも当てはまるカイ二乗分布をモーメント法により求めることでkappaを推定する

    return kappa
'''

def detection(database,c,kappa):
    '''
    系列を方向データとして扱えるように整形して、異常検知を行う
    系列を方向データに変換するには、sequence2Vector()関数を用いる
    この時に長さを１に正規化して方向データとする

    異常度を1-np.dot(mean,x)として定めると、これがカイ二乗分布に従うことがわかっているので
    棄却域を自由度とkappaから決めることで異常判定する
    '''

    anomaly_list = []
    directional_Database = []
    size = len(database)
    for i in range(size):
        directional_Database.append( sequence2Vector(database[i],c) )

    '''
    meanは最尤推定できるがkappaは解析的に求めることはできないので、訓練データから
    もっとも当てはまりの良いkappaをモーメント法により求めてその値を推定値とする
    現在はkappaを決めうちで使っている

    kappa = learn_Kappa(derectional_database,mean)
    '''

    mean = np.mean(directional_Database,axis = 0)
    norm = np.linalg.norm(mean)
    mean /= norm
    
    rv = chi2(df=c**2-1,scale=1/(2*kappa))
    eps = rv.ppf(0.99)
    print(eps)

    for i in range(size):
        score = 1 - np.dot(mean,directional_Database[i])
        if score > eps:
            print(i)
            print(score)
            anomaly_list.append(i)

    return anomaly_list


c=5
sequence = [0,0,0,0]
database = [[0,0,0,0],[0,0,0,0,0,0,0],[0,0],[0,0,0,0,0],[1,2,3,4,4,4,4,4,4]]
print(sequence2Vector(sequence,c))
print(detection(database,c,100))