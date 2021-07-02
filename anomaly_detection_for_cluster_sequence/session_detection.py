def make_Table(database,c):
    
    '''
    cは状態数
    ログのクラスタリングから得られる系列の集合databaseから、各状態の出現回数と遷移を記憶する。
    生成するtableは、c * (c + 1)の行列とする。
    (i,j)成分はstate_iからstate_jへの遷移の回数を保存する。ただし、(i,c + 1)成分はstate_iからの遷移の回数を保存する．
    例えば、系列の集合Sを次のようにすると
        S = {00120, 120, 2201, 0321,}
        table = {1  2   0   1   4
                0   0   2   0   2
                3   1   1   0   5
                0   0   1   0   1
                }
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

def make_ScoreTable(table,c):
    '''
    遷移のスコアからなるtableを作成する
    高倉先生論文でのp(・)q(・)をこの表にアクセスすることで得られる．
    '''

    score_table = [[0] * c for i in range(c)]
    n_transition = 0

    for i in range(c):
        n_transition += table[i][c]
    '''
    for i in range(c):
        for j in range(c):
            p_table[i][j] = table[i][j] / n_transition

    for i in range(c):
        for j in range(c):
            q_table[i][j] = table[i][j] / table[i][c]
    '''

    for i in range(c):
        for j in range(c):
            if table[i][c] == 0:
                score_table[i][j] = 0
            else:
                score_table[i][j] = (table[i][j] ** 2) / (table[i][c] * n_transition)
            #score_table[i][j] = (table[i][j])**2 / (table[i][c] * n_transition)


    return score_table

def seq_To_Scores(l,seq,score_table):

    length = len(seq)
    transition_scores = [0]*(length-1)
    session_scores = [0] * length

    for i in range(length-1):
        '''
        i番目の遷移のスコアリングを行い，リストとしてまとめる
        '''
        transition_scores[i] = score_table[ seq[i] ][ seq[i+1] ]

    for i in range(length):
        '''
        i番目のログをスコアリングする
        スコアリングはr番目の遷移からs番目の遷移の合計とする．
        r = max(0,i-l)
        s = min(length-1,i+l-1)
        '''
        r = max(0,i-l+1)
        s = min(length-1,i+l)
        session_scores[i] = sum(transition_scores[r:s])

    return session_scores


def detection(c_list,s_list,session_index,c,eps,l=5):
    '''
    client-ipでのリストc-list
    server-ipでのリストs-list
    の２つからスコアのリストとしてそれぞれ変換したい．
    ここで，長さがl未満のシーケンスはスコアリングに用いられないので，このシーケンスに対応するスコアのリストは[]としている．

    sessionのindexは，[[c_listでの番号，index],[s_listでの番号，index]]で与えられる    
    '''
    database = c_list
    table = make_Table(database,c)
    score_table = make_ScoreTable(table,c)
    c_score_list = []
    n_session = len(session_index)
    session_scores = [0] * n_session
    anomaly_sessions = []
    c_length = len(c_list)

    for i in range(c_length):
        if len(c_list[i]) < l:
            c_score_list.append([])
        else:
            c_score_list.append( seq_To_Scores(l,c_list[i],score_table) )

    
    for i in range(n_session):
        '''
        i番目のセッションのスコアリングを行う
        '''
        if c_score_list[session_index[i][0][0]] != []:
            score1 = c_score_list[session_index[i][0][0]][session_index[i][0][1]]
            session_scores[i] = score1
        else:
            session_scores[i] = 1000000

    for i in range(len(eps)):
        anomaly_sessions = list( filter(lambda x:session_scores[x] < eps[i],range(n_session) ) ) 
    
    print(anomaly_sessions)
    
    return session_scores
    
def cal_DetectionTabale(session_scores,eps,label):
    '''
    正常データのラベルは0
    攻撃データのラベルは1
    として検知率と誤検知率を計算する

    閾値epsはこの関数に関しては１つの値を与えることとする
    正解ラベルはlabel，各セッションのスコアはsession_scoresで与えられる
    TPなどの表の要素は高倉論文CSS2015を参考にする．
    '''

    n_normal = label.count(0)
    n_anomaly = label.count(1)
    tp= 0
    tn = 0
    accuracy_table = [[0,0],[0,0]]

    anomaly_list = list( filter(lambda x:session_scores[x] < eps,range(len(session_scores) ) ) )
    for i in range(len(session_scores)):
        if (i in anomaly_list) and label[i] == 1:
            tp += 1
        elif not(i in anomaly_list) and label[i] == 0:
            tn += 1

    accuracy_table[0][0] = tp
    accuracy_table[1][1] = tn
    accuracy_table[0][1] = n_normal - tn
    accuracy_table[1][0] = n_anomaly - tp

    dr = tp/(tp+accuracy_table[1][0])
    fpr = (accuracy_table[0][1])/(accuracy_table[0][1]+tn)

    return [accuracy_table,(dr,fpr)]

c = 3
database = [[0,0,1],[1,2,0]]
session_index = [ [[0,0],[0,0]] , [[1,0],[0,0]] ,[ [0,1],[0,0]] , [[0,2],[0,0]] , [[1,1],[0,0]] ,[[1,2],[0,0]] ]
table = make_Table(database,c)
session_scores = detection(database,[],session_index,c,[0.0])
actable = cal_DetectionTabale(session_scores,0.0,[0,0,0,0,1,1])[0]
print(actable)