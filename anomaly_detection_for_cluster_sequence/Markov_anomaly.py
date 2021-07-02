import math

def make_table(database,c):
    
    '''
    cは状態数
    ログのクラスタリングから得られる系列の集合databaseから、各状態の出現回数と遷移を記憶する。
    生成するtableは、c * (c + 1)の行列とする。
    (i,j)成分はstate_iからstate_jへの遷移の回数を保存する。ただし、(i,c + 1)成分はstate_iからの遷移の回数を保存する

    例えば、系列の集合Sを次のようにすると
        S = {00120, 120, 2201, 0321,}
        table = {1  2   0   1   4
                0   0   2   0   2
                3   1   1   0   5
                0   0   1   0   1}
    tableは上のようになる。
    '''

    size = len(database)
    table = [[0] * (c + 1) for i in range(c)]
    for i in range(size):
        sequence = database[i]
        length = len(sequence)
        for j in range(length - 1):
            table[ sequence[j] ][ sequence[j+1] ] += 1
            table[ sequence[j] ][c] += 1

    return table

def cal_Frequency(state1,state2,table):
    
    '''
    系列に出現する状態やその頻度をリストに記憶しておく
    そして、そこから各遷移の頻度を求める。それが、マルコフ連鎖における遷移確率の推定値になる。
    state1 -> state2の遷移確率を返す関数 
    '''
    c = len(table)
    frequency = 0
    if table[state1][c] != 0:
        frequency = (table[state1][state2] / table[state1][c])

    return frequency

def cal_Anomaly_Score(sequence,table):
    ##sequenceのanomaly_scoreを計算する関数
    
    score = 0
    length = len(sequence)
    if length == 1:
        return 0;
    for i in range( length - 1 ):
        frequency = cal_Frequency(sequence[i],sequence[i + 1],table)
        score += math.log2(frequency)
    score /= (length-1)

    return score*(-1)

def detection(database,c,eps):
    '''
    databaseに含まれる系列の異常検知をする
    cal_Anomaly_Score関数でスコアリングして閾値epsよりも大きい系列が異常と判定されることになる
    異常と判定された系列のdatabase上でのインデックスのリストを返す
    まずはからのリストanomaly_listを生成してから、appendで付け加えていくことをする
    '''

    
    table = make_table(database,c)
    size = len(database)
    scores = [0] * size
    n_eps = len(eps)
    
    for i in range(size):
        seq = database[i]
        scores[i] = cal_Anomaly_Score(seq,table)
    print(scores)
    print("---------------")

    for i in range(n_eps):
        anomaly_list = []
        for j in range(size):
            if scores[j] > eps[i]:
                anomaly_list.append(j)
        print(anomaly_list)
        print("---------------")
    return scores

def cal_F_Score(databese,labels,anomaly_list):
    '''
    異常検知プログラムの結果から，F値を計算する
    F値の計算には，正常標本精度と異常標本精度の二つを求めることが必要になる
    anomaly_listには，異常と判定されたdataのリストが含まれている．
    labelsにはデータが本当は正常か異常かのラベルが保存されている

    r1,r2はそれぞれ正常標本精度と異常標本精度を保存する
    '''
    size = len(database)
    r1 = 0
    r2 = 0
    accuracy_table = [0]*4
    for i in range(size):
        if not(i in anomaly_list) and labels[i] == 0:
            r1 += 1
        elif (i in anomaly_list) and labels[i] == 1:
            r2 += 1
    n_normal = labels.count(0)
    n_anomaly = labels.count(1)
    print(r1,r2)
    accuracy_table[0] = r1
    accuracy_table[1] = n_anomaly - r2
    accuracy_table[2] = n_normal - r1
    accuracy_table[3] = r2
    print(accuracy_table)

    r1 /= n_normal
    r2 /= n_anomaly
    f = 2*r1*r2 / (r1 + r2)
    
    return f

c = 4
database = [[0,0,1,2,0],[1,2,0],[2,2,0,1],[0,3,2,1]]
labels = [0,0,1,1]
table = make_table(database,c)
print(table)
detection(database,c,[1.0])