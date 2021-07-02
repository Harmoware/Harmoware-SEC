
def lcs(sequence1,sequence2):
    ##sequence1とsequence2の最長共通部分列の長さを返す関数
    len1 = len(sequence1) 
    len2 = len(sequence2)
    table = [[0] * (len2+1) for i in range(len1+1)]

    for i in range(len1+1):
        for j in range(len2+1):
            if i == 0 or j == 0:
                table[i][j] = 0
            else:
                if sequence1[i-1] == sequence2[j-1]:
                    table[i][j] = table[i-1][j-1] + 1
                else:
                    table[i][j] = max(table[i-1][j],table[i][j-1])

    return table[len1][len2]

def cal_Similarity_LCS(sequence1,sequence2):
    ##二つの系列の類似度を求める。定義式はサーベイ論文の(1)に書いてある通り
    nLCS = lcs(sequence1,sequence2) / (len(sequence1) * len(sequence2)) ** 0.5
    
    return nLCS

def cal_Similarity_Ngram(sequence1,sequence2,n):
    '''
    共通するn-gramの個数を元にして類似度を計算する
    cは共通するtokenの個数を保存する変数
    '''

    c = 0
    length1 = len(sequence1)
    length2 = len(sequence2)
    add = ['|'] * (n-1)
    sequence1.append(add)
    sequence2.append(add)
    sequence1 = add + sequence1
    sequence2 = add + sequence2
    
    for i in range(length1 + n-1):
        for j in range(length2 + n -1):
            if sequence1[i:i+n-1] == sequence2[j:j+n-1]:
                c += 1

    return c /(length1 + length2 + 2*(n-1) - c)


def score_Anomaly_LCS(sequence,database):
    '''
    データベース上にある系列全てに対して類似度を求めてその平均を出す。その逆数が考えている系列のanomaly-scoreである。
    databaseは、系列のリストとして与えられるとする

    自分以外の系列との類似度が0のものがあったら、どうするかを考えなければならない
    その場合の処理は、かなり大きい値を当てるか特別な値を決めて割り当てる
    '''

    score = 0
    for s in database:
        score += cal_Similarity_LCS(s,sequence)
        if s == sequence:
            score -= 1

    score /= (len(database) - 1)

    if score == 0:
        return 100000
    return (1.0 / score)

def score_Anomaly_Ngram(sequence,database,n):

    score = 0
    for s in database:
        score += cal_Similarity_Ngram(s,sequence,n)
        if s == sequence:
            score -= 1

    score /= (len(database) - 1)

    if score == 0:
        return 100000
    return (1.0 / score)

def detection_LCS(database,eps):
    
    '''
    databaseの系列を全てスコアリングしてから異常なものを閾値epsを用いて検知する
    epsの値はヒューリスティックに決める
    返す値としては、異常な系列のインデックスのリストを返す

    閾値のリストeps[]を受け取り，それぞれの閾値で異常検知した結果を個別に出力する
    scores[]には各シーケンスのスコアが保存される

    anomaly_list = list(filter(lambda x:scores[x] > eps[i], range(size)))
    とかでもいいかもしれない．見やすい
    内包表記なども書いてみるとかなりコードがコンパクトになる
    '''
    
    index = [i for i in range(len(database))]
    scores = [score_Anomaly_LCS(database[i],database) for i in range(len(database))]
    print(scores)

    for i in range(len(eps)):
        anomaly_list = [t for t in range(len(scores)) if scores[t] > eps[i]]
        
        print(anomaly_list)
        print("-----------------")

    return scores

def detection_Ngram(database,eps,n):

    size = len(database)
    n_eps = len(eps)
    scores = [0] * size
    for i in range(size):
        scores[i] = score_Anomaly_Ngram(database[i],database,n)
    print(scores)

    for i in range(n_eps):
        anomaly_list = []
        for j in range(size):
            if scores[j] > eps[i]:
                anomaly_list.append(j)
        print(anomaly_list)
        print("-----------------")

   

def cal_F_Score(databese,labels,anomaly_list):
    '''
    異常検知プログラムの結果から，F値を計算する
    F値の計算には，正常標本精度と異常標本精度の二つを求めることが必要になる
    anomaly_listには，異常と判定されたdataのリストが含まれている．
    labelsにはデータが本当は正常か異常かのラベルが保存されている

    r1,r2はそれぞれ正常標本精度と異常標本精度を保存する
    accuracy_tableは分類結果と正解の表を表す
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
    accuracy_table[0] = r1
    accuracy_table[1] = n_anomaly - r2
    accuracy_table[2] = n_normal - r1
    accuracy_table[3] = r2
    print(accuracy_table)

    r1 /= n_normal
    r2 /= n_anomaly
    if r1 == 0 and r2 == 0:
        return 0

    f = 2*r1*r2 / (r1 + r2)

    return f
    
database = [[0,0,1,2,0],[1,2,0],[2,2,0,1],[0,3,2,1],[1,1,1,1,1,1,1,1,1,1]]
labels = [1,1,1,1,0]
detection_LCS(database,[0,0])