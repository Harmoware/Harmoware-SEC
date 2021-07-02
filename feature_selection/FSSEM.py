
# coding: utf-8

# In[1]:

import sklearn.mixture as mix
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








def score_likelihood(data, GM):   
    '''
    the higher the betterな尤度基準の評価値を返す
    bicやaicならばthe lower the better， score(普通の対数尤度)ならばthe higher the betterであることに注意
    
    入力
        data:分析対象のデータ
        GM:混合ガウス分布のモデル
    出力
        score:評価値(ML基準)
    '''
    score = -GM.bic(data)    # score?
    return score
    






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







def min_dist(weights, means, covs):
    '''
    全クラスタペアに対して距離関数を計算し、最も近いクラスタペアを求める
    bouman_merge用
    入力
        weights : 各クラスタの確率(size:clusters)
        means : 各クラスタの平均(size:clusters, dim)
        covs : 各クラスタの共分散行列(size:clusters, dim, dim)
    出力
        cluster_pair：最も「近い」クラスタペア[i,j]
        w_best : i,jをmergeしたクラスタの確率
        m_best : i,jをmergeしたクラスタの平均(size:dim)
        c_best : i,jをmergeしたクラスタの共分散行列(size:dim, dim)
    '''
    clusters = weights.shape[0]
    dim = means.shape[1]
    
    dist_minimum = np.inf
    cluster_pair = [-1, -1]
    w_best = 0
    m_best =  np.zeros(dim)
    c_best = np.zeros((dim, dim))
    
    for j in range(1, clusters):
        for i in range(j):
            
            w_i = weights[i]
            w_j = weights[j]
            w_ij = w_i + w_j
            
            m_i = means[i]
            m_j = means[j]
            m_ij = (w_i*m_i + w_j*m_j)/w_ij
            diff_m_i_ij = np.matrix(m_i - m_ij)
            diff_m_j_ij = np.matrix(m_j - m_ij)
            
            c_i = covs[i]
            c_j = covs[j]
            c_ij = (w_i*(c_i + diff_m_i_ij.T * diff_m_i_ij) + w_j*(c_j + diff_m_j_ij.T * diff_m_j_ij)) / w_ij
            
            det_c_i = np.linalg.det(c_i)
            det_c_j = np.linalg.det(c_j)
            det_c_ij = np.linalg.det(c_ij)
            
            # 対数の底は2? e?
            dist_temp = w_i*np.log(det_c_ij/det_c_i) + w_j*np.log(det_c_ij/det_c_j)
            
            if dist_temp <= dist_minimum:
                dist_minimum = dist_temp
                cluster_pair = [i, j]
                w_best = w_ij
                m_best = copy.deepcopy(m_ij)
                c_best = copy.deepcopy(c_ij)
    
    return cluster_pair, w_best, m_best, np.array(c_best)







def bouman_merge(weights, means, covs):
    '''
    最も近いクラスタペアをmergeし，そのときのクラスタの状態を計算
    em_clustering用
    入力
        weights : 各クラスタの確率(size:<clusters>)
        means : 各クラスタの平均(size:<clusters>, dim)
        covs : 各クラスタの共分散行列(size:<clusters>, dim, dim)
    出力
        w_new : i,jをmergeした際の各クラスタの確率(size:<clusters>-1)
        m_new : i,jをmergeした際の各クラスタの平均(size:<clusters>-1, dim)
        c_new : i,jをmergeした際の各クラスタの共分散行列(size:<clusters>-1, dim, dim)
        precisions_chol : c_newの各cholesky分解の逆行列のリスト(size:<clusters>-1, dim, dim)
    '''
    dim = means.shape[1]
    
    cluster_pair, w_merged, m_merged, c_merged = min_dist(weights, means, covs)
    w_new = np.append(np.delete(weights, cluster_pair, axis=0), [w_merged], axis=0)
    m_new = np.append(np.delete(means, cluster_pair, axis=0), [m_merged], axis=0)
    c_new = np.append(np.delete(covs, cluster_pair, axis=0), [c_merged], axis=0)
    
    precisions_chol = np.empty(c_new.shape)
    for i, covariance in enumerate(c_new):
        try:
            cov_chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol[i] = sp.linalg.solve_triangular(cov_chol, np.eye(dim), lower=True).T
    
    return w_new, m_new, c_new, precisions_chol







def em_clustering(data):
    '''
    与えられたdataをEMクラスタリングする。結果のlabelを返す
    クラスタ数の判定にBoumanのアルゴリズムを用いている
    入力
        data : データ(size:num, dim)
    出力
        label_best : EMクラスタリングで求めた最適なラベル割当(size:num)
        clusters_best : Boumanのアルゴリズムで求めた最適なクラスタ数
        GM_best : 最適なGMモデル
    '''
    
    num = data.shape[0]
    dim = data.shape[1]
    
    # clusters_maxの設定は適切か?
    clusters_max =  int(num**(1/4))+1
    
    GM_best = mix.GaussianMixture(n_components = clusters_max)
    GM_best.fit(data)
    label_best = GM_best.predict(data)
    clusters_best = clusters_max
    score_em_best = score_likelihood(data, GM_best)
    print('score_em:',score_em_best)
    
    GM_temp = copy.deepcopy(GM_best)
    
    for i in range(1, clusters_max):
        print('loop_Bouman:', i)
        w_temp, m_temp, c_temp, pre_chol_temp                     = bouman_merge(GM_temp.weights_, GM_temp.means_, GM_temp.covariances_)
        
        GM_temp = mix.GaussianMixture(n_components = clusters_max-i)
        GM_temp.weights_ = w_temp
        GM_temp.means_ = m_temp
        GM_temp.covariances_ = c_temp
        GM_temp.precisions_cholesky_ = pre_chol_temp
        
        score_em_temp = score_likelihood(data, GM_temp)    # score?
        print('score_em:', score_em_temp)
        if score_em_temp >= score_em_best:
            label_best = GM_temp.predict(data)
            clusters_best = clusters_max-i
            score_em_best = score_em_temp
            GM_best = copy.deepcopy(GM_temp)
    
    print('score_em_best:', score_em_best)
    return (label_best, clusters_best, GM_best)







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
        GM_best : 特徴量feature_bestのときのGMモデル
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
    GM_best = mix.GaussianMixture()
    
    # data_confirmed : 既に選択されたfeatureに関するdata subset
    data_confirmed = data[:, feature_selected[:] != 0]
    
    for i in range(dim):
        print('loop_fs:', i)
        if feature_selected[i] == 0:
            data_temp = np.c_[data_confirmed, data[:, i]]
            
            label_temp, clusters_temp, GM_temp = em_clustering(data_temp)
            if criterion_mode == 'ML':
                score_temp = score_likelihood(data_temp, GM_temp)
                print('score_ml:', score_temp)
            elif criterion_mode == 'SS':
                score_temp = score_ss(data_temp, label_temp, clusters_temp)
                print('score_ss:', score_temp)
            
            if score_temp > score_best:
                feature_best_index = i
                label_best = copy.deepcopy(label_temp)
                clusters_best = clusters_temp
                score_best = score_temp
                GM_best = copy.deepcopy(GM_temp)
    
    feature_best[feature_best_index] = num_loop+1
    
    return {'FEATURE':feature_best, 'LABEL':label_best, 'CLUSTER':clusters_best, 'SCORE':score_best, 'GM':GM_best}







def fssem(data, *, eps=0, criterion_mode='ML'):
    '''
    FSSEM(EMを用いた特徴選択)を実行
    入力
        data : 分析対象のデータ(size:num, dim)
        eps : FSSEMでのループが終了する条件の閾値
        criterion_mode={'ML', 'SS'} : どの基準で評価するか
    
    出力
        state_best
            FEATURE : 特徴量の選択とその優先度を表すリスト(size:dim)
            LABEL : 特徴量feature_bestのときのクラスタ割当(size:num)
            CLUSTER : 特徴量feature_bestのときのクラスタ数
            SCORE : 特徴量feature_bestのときの評価値
            GM : 特徴量feature_bestのときのGMモデル
    '''
    
    data = scale(data)
    
    num = data.shape[0]
    dim = data.shape[1]
    
    state_best = {'FEATURE':np.zeros(dim), 'LABEL':np.empty(num), 'CLUSTER':0, 'SCORE':0.0, 'GM':mix.GaussianMixture()}
    state_temp = {'FEATURE':np.zeros(dim), 'LABEL':np.empty(num), 'CLUSTER':0, 'SCORE':0.0, 'GM':mix.GaussianMixture()}
    
    for loop in range(dim):
        
        state_temp = feature_selection(data, state_temp['FEATURE'], criterion_mode, num_loop=loop)
        print("loop, state_temp:", loop, state_temp)
        
        if loop == 0:
            state_best = copy.deepcopy(state_temp)
        else:
            # S1, C1: state_tempのfeature subset, cluster assignment
            # S2, C2: state_bestのfeature subset, cluster assignment
            
            if criterion_mode == 'ML':
                
                def gen_GM(data_prev, pi_prev, mu_prev, sigma_prev, data_next):
                    # num = data_t.shape[0]
                    clusters = mu_prev.shape[0]
                    dim_prev = data_prev.shape[1]
                    dim_next = data_next.shape[1]
                    
                    def pdf_Gauss(x, mu, sigma):
                        a = ((2*np.pi)**dim_prev) * np.linalg.det(sigma)
                        
                        diff_x = np.matrix(x-mu)
                        sigma_inv = np.matrix(sigma).I
                        b = -0.5 * diff_x * sigma_inv * diff_x.T    # bは(1,1)行列になっている
                        
                        return np.exp(b[0,0]) / np.sqrt(a)
                    
                    E = np.empty((num, clusters))
                    for i in range(num):
                        temp = [pdf_Gauss(data_prev[i], mu_prev[s], sigma_prev[s]) * pi_prev[s] for s in range(clusters)]
                        sum_temp = sum(temp)
                        E[i] = np.array([temp[j] / sum_temp for j in range(clusters)])
                    
                    
                    
                    pi = sum(E) / num
                    
                    mu = np.empty((clusters, dim_next))
                    for j in range(clusters):
                        sum_temp_mu = 0
                        for i in range(num):
                            sum_temp_mu += E[i][j] * data_next[i]
                        mu[j] = sum_temp_mu / (num * pi[j])
                    
                    sigma = np.empty((clusters, dim_next, dim_next))
                    for j in range(clusters):
                        sum_temp_sigma = 0
                        for i in range(num):
                            diff_x = np.matrix(data_next[i]-mu[j])
                            sum_temp_sigma += np.array(E[i][j] * diff_x.T * diff_x)
                        sigma[j] = sum_temp_sigma / (num * pi[j])
                    
                    precisions_chol = np.empty((clusters, dim_next, dim_next))
                    for i, covariance in enumerate(sigma):
                        try:
                            cov_chol = np.linalg.cholesky(covariance)
                        except np.linalg.LinAlgError:
                            raise ValueError(estimate_precision_error_message)
                        precisions_chol[i] = sp.linalg.solve_triangular(cov_chol, np.eye(dim_next), lower=True).T
                    
                    GM = mix.GaussianMixture(n_components = clusters)
                    GM.weights_ = pi
                    GM.means_ = mu
                    GM.covariances_ = sigma
                    GM.precisions_cholesky_ = precisions_chol
                    
                    return GM
                
                
                # S1_C1: temp, S_2_C1と積をとる
                # S2_C2: best, S_1_C2と積をとる
                GM_S2_C1 = gen_GM(data[:, state_temp['FEATURE'][:] != 0], state_temp['GM'].weights_, state_temp['GM'].means_,                                                               state_temp['GM'].covariances_, data[:, state_best['FEATURE'][:] != 0])
                GM_S1_C2 = gen_GM(data[:, state_best['FEATURE'][:] != 0], state_best['GM'].weights_, state_best['GM'].means_,                                                               state_best['GM'].covariances_, data[:, state_temp['FEATURE'][:] != 0])
                
                score_S2_C1 = score_likelihood(data[:, state_best['FEATURE'][:] != 0], GM_S2_C1)
                score_S1_C2 = score_likelihood(data[:, state_temp['FEATURE'][:] != 0], GM_S1_C2)
            
            elif criterion_mode == 'SS':
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

