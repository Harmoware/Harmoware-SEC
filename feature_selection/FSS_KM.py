
# coding: utf-8

# In[ ]:

import sklearn.mixture as mix
import sklearn.cluster as clust
import scipy as sp
import numpy as np
import copy

'''
num:データの個数
dim:データの特徴量次元
state = {'FEATURE', 'LABEL', 'CLUSTER', 'SCORE', 'GM'}:ディクショナリ
feature:選択した特徴量を表すリスト
label:データをクラスタリングした際のラベル
clusters:データをいくつのクラスタに分類するか。Boumanのアルゴリズムによって求める。
score:評価値
'''


def scale(data):
    num = data.shape[0]
    dim = data.shape[1]
    
    # 属性ごとに平均値と標準偏差を計算
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    
    # 属性ごとにデータを標準化
    data = np.array([[(data[j,i]-mu[i])/sigma[i] for i in range(dim)] for j in range(num)])
    
    return data








def score_ss(data, label, clusters):   # クラスタリング結果の評価関数
    '''
    クラス間分散Sbとクラス内分散Swを求めて
    trace(Sw^{-1}Sb)を評価値として返す。
    
    入力
        data:分析対象のデータ
        label:分析対象の分類ラベル
        clusters:分類するクラスタ数
    出力
        score:評価値(SS基準)
    '''
    
    num = data.shape[0]
    dim = data.shape[1]
    
    Sw = np.zeros((dim,dim))              # クラス内共分散行列
    Sb = np.zeros((dim,dim))              # クラス間共分散行列
    
    # クラスタ毎に分けたデータセット
    subdata = np.array([data[label[:]==i, :] for i in range(clusters)])
    
    # 各クラスタにデータが入る確率(データ割合)
    pi = np.array([subdata[i].shape[0]/num for i in range(clusters)])
    
    for i in range(clusters):
        if subdata[i].shape[0] != 1 or subdata[i].shape[1] != 1:
            Sw += pi[i] * np.cov(subdata[i], rowvar=0)
    
    mean_all = np.mean(data, axis=0)    # 全体の平均ベクトル
    for i in range(clusters):
        mean_diff = np.matrix(np.mean(subdata[i], axis=0) - mean_all)
        Sb += pi[i] * mean_diff.T * mean_diff
    
    try:                                            # trace(Sw^{-1}Sb)の計算
        score = np.trace(np.linalg.solve(Sw, Sb))
    except np.linalg.LinAlgError:                # Sw^{-1}Sbが計算できない場合
        print("!!! LinAlgError !!!")
        '''
        sb_scalar = sw_scalar = 0
        for i in range(clusters):
            sb_scalar += pi[i]*numpy.linalg.norm(means[i] - mean_all)
        for i in range(num):
            j = label[i]
            sw_scalar += pi[j]*numpy.linalg.norm(data[i]-means[j]) / N_k[j]
        score = sb_scalar / sw_scalar
        '''
    
    return score







def km_clustering(data):
    '''
    与えられたdataをKMeansでクラスタリングする。結果のlabelを返す
    入力
        data : データ(size:num, dim)
    出力
        label_best : KMeansクラスタリングで求めた最適なラベル割当(size:num)
        clusters_best : 最適なクラスタ数
        Model_best : 最適なKMeansモデル
    '''
    
    num = data.shape[0]
    dim = data.shape[1]
    
    # clusters_maxの設定は適切か?
    clusters_max =  int(num**(1/4))+1
    
    Model_best = clust.KMeans(n_clusters = clusters_max)
    Model_best.fit(data)
    label_best = Model_best.labels_
    clusters_best = clusters_max
    score_km_best = score_ss(data, label_best, clusters_best)
    print('score_km:',score_km_best)
    
    for i in range(1, clusters_max):
        print('loop_KMeans:', i)
        
        Model_temp = clust.KMeans(n_clusters = clusters_max-i)
        Model_temp.fit(data)
        label_temp = copy.deepcopy(Model_temp.labels_)
        score_km_temp = score_ss(data, label_temp, clusters_max-i)    # score?
        
        print('score_km:', score_km_temp)
        if score_km_temp >= score_km_best:
            label_best = label_temp
            clusters_best = clusters_max-i
            score_km_best = score_km_temp
            Model_best = copy.deepcopy(Model_temp)
    
    print('score_km_best:', score_km_best)
    return (label_best, clusters_best, score_km_best, Model_best)







def feature_selection(data, feature_selected, criterion_mode, *, num_loop=0):
    '''
    relevantな特徴量を選択する。逐次順方向探索を採用
    入力
        data : データ(size:num, dim)
        feature_selected : 前回までのループですでに選択された特徴量(size:num_loop)
        criterion_mode : 'SS' or 'ML'
        num_loop : 特徴選択の実行回数
    出力
        state
        feature_best : 特徴量の選択とその優先度を表すリスト(size:dim)
        label_best : 特徴量feature_bestのときのクラスタ割当(size:num)
        clusters_best : 特徴量feature_bestのときのクラスタ数
        score_best : 特徴量feature_bestのときの評価値
        Model_best : 特徴量feature_bestのときのモデル
    '''
    
    num = data.shape[0]
    dim = data.shape[1]
    
    # feature_best_index : 最良の評価値のときに新たに選択されるfeatureのindex
    #　**_best : このループでの最良の評価値を出すセット
    feature_best_index = -1
    feature_best = copy.deepcopy(feature_selected)
    label_best = np.empty(num)
    clusters_best = 0
    score_best = - np.inf
    Model_best = clust.KMeans()
    
    # data_confirmed : 既に選択されたfeatureに関するdata subset
    data_confirmed = data[:, feature_selected[:] != 0]
    
    for i in range(dim):
        print('loop_fs:', i)
        if feature_selected[i] == 0:
            data_temp = np.c_[data_confirmed, data[:, i]]
            
            label_temp, clusters_temp, score_temp, Model_temp = km_clustering(data_temp)
            
            if score_temp > score_best:
                feature_best_index = i
                label_best = copy.deepcopy(label_temp)
                clusters_best = clusters_temp
                score_best = score_temp
                Model_best = copy.deepcopy(Model_temp)
    
    feature_best[feature_best_index] = num_loop+1
    
    return {'FEATURE':feature_best, 'LABEL':label_best, 'CLUSTER':clusters_best, 'SCORE':score_best, 'MODEL':Model_best}







def fss_km(data, *, eps=0, criterion_mode='SS'):
    '''
    FSS_KM(KMeansを組み合わせた特徴選択)を実行
    入力
        data : 分析対象のデータ(size:num, dim)
        eps : FSS_KMでのループが終了する条件の閾値
        criterion_mode='SS' : SS基準で評価
    
    出力
        state_best
            FEATURE : 特徴量の選択とその優先度を表すリスト(size:dim)
            LABEL : 特徴量feature_bestのときのクラスタ割当(size:num)
            CLUSTER : 特徴量feature_bestのときのクラスタ数
            SCORE : 特徴量feature_bestのときの評価値
            MODEL : 特徴量feature_bestのときのKMeansモデル
    '''
    
    data = scale(data)
    
    num = data.shape[0]
    dim = data.shape[1]
    
    state_best = {'FEATURE':np.zeros(dim), 'LABEL':np.empty(num), 'CLUSTER':0, 'SCORE':0.0, 'MODEL':clust.KMeans()}
    state_temp = {'FEATURE':np.zeros(dim), 'LABEL':np.empty(num), 'CLUSTER':0, 'SCORE':0.0, 'MODEL':clust.KMeans()}
    
    for loop in range(dim):
        
        state_temp = feature_selection(data, state_temp['FEATURE'], criterion_mode, num_loop=loop)
        print("loop, state_temp:", loop, state_temp)
        
        if loop == 0:
            state_best = copy.deepcopy(state_temp)
        else:
            # S1, C1: state_tempのfeature subset, cluster assignment
            # S2, C2: state_bestのfeature subset, cluster assignment
            
            score_S2_C1 = score_ss(data[:, state_best['FEATURE'][:] != 0], state_temp['LABEL'], state_temp['CLUSTER'])
            score_S1_C2 = score_ss(data[:, state_temp['FEATURE'][:] != 0], state_best['LABEL'], state_best['CLUSTER'])
            
            normal_v = state_temp['SCORE'] * score_S2_C1
            normal_v_best = state_best['SCORE'] * score_S1_C2
            
            diff_score = normal_v - normal_v_best
            
            print('difference:', diff_score)
            
            if diff_score > 0:
                state_best = copy.deepcopy(state_temp)
            else:
                break
        
        print(state_best['FEATURE'])
    
    return state_best

