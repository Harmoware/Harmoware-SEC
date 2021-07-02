import re#正規表現のlibrary
import pandas as pd#pandas:構造データ用のlibrary
import numpy as np
from sklearn.externals import joblib#オブジェクトの保存・読み込みに利用
import warnings;warnings.simplefilter('ignore', DeprecationWarning)#非推奨関数に対する警告を抑える
from time import time#時間計測
from urllib import parse#urlの解析に使えるライブラリ
from os.path import splitext
from os.path import exists#ファイルの有無を確認するのに仕様
import random

#matchした部分全部を表示するメソッド
def re_display(r):
    """
        入力:r:正規表現
        出力:rの中のマッチした部分を全列挙
    """
    for (key,para) in zip(tuple(r.groupdict().keys()),r.groups()):#matchの中で名付けたパートの羅列
        print("%s = %s" % (key,para),end="       ")
    print()

#port番号を含まないものには対応していない
    #col_of_DFを変えれば対応可能だが、出力するDFの形が変わるので影響する部分全てに対応が必要なため非推奨
def Make_PlaneDF_from_log(logname):
    """
        機能:
            読み込んだログファイルを各成分毎にparseだけしたDFを作成する
            ここで作成したDFを元に入力用のDFなどを計算する
            保存場所は"./DFs/DF_(読み込んだログ名)"
        引数の説明:
            logname:元のログファイルの名前(正確には./Accesslogs/以下のパス)
        特徴量の説明:
            day,month,year,hour,minute,second:ログの時刻
            zero_time:ログを取得したところの基準時刻
            response_time:応答までの時間(ミリ秒)
            client_ip:ユーザのipアドレス
            port:接続ポート
            request_status:リクエストがどうなったか．例:TCP_MISS
            status_to_client:ステータスコード．例:200,403
            size_of_reply:応答のサイズ
            request_method:リクエストされたメソッド．例:GET/POST
            url_from_client:ユーザが指定したurl
            server_ip:上のurlが指すipアドレス
            user_name:ユーザの名前(今のところ空欄が多い)
            hierarchy_status:(理解できなかったので公式の文をそのまま)Squid hierarchy status. 例:DEFAULT_PARENT 
            server_peer_name:通信したサーバの名前
            MIME_content_type:(理解できなかったので公式の文をそのまま)MIME content type
            Referer:リクエストurl
            User_Agent:ユーザの環境(OSやブラウザの情報)
    """
    col_of_DF = ['day', 'month', 'year', 'hour', 'minute', 'second', 'zero_time', 'response_time', 'client_ip', 'port', 'request_status', 'status_to_client', 'size_of_reply', 'request_method', 'url_from_client', 'server_ip', 'user_name', 'hierarchy_status', 'server_peer_name', 'MIME_content_type', 'Referer', 'User_Agent']
    main_DF = pd.DataFrame(columns=col_of_DF)#空のDataFrameを作成,後にログからparseしたデータを入れる
    logdir = "./Accesslogs/"+logname
    dfdir = "./DFs/DF_"+logname#main_DFの保存先のパス，まずは保存ディレクトリを変更
    f = open(logdir)#ログファイルを読み込み
    i = 0#ログをDataFrameに直すときのindexに利用
    for line in f:
        #下の行でマッチのパターンを定義
        #row文字列を複数行で書く場合は 1.途中でインデントをいれないこと 2.改行したときにバックスラッシュを入れる
        r = re.match(r"""^(?P<day>[^/]*)/*(?P<month>[^/]*)/*(?P<year>[^:]*):(?P<hour>[^:]*):(?P<minute>[^:]*):(?P<second>[^\s]*)\s*(?P<zero_time>[^\s]*)\s*(?P<response_time>[^\s]*)\s*(?P<client_ip>[^\s]*)\s*(?P<port>[^\s]*)\s(?P<request_status>[^/]*)/(?P<status_to_client>[^\s]*)\s*(?P<size_of_reply>[^\s]*)\s*(?P<request_method>[^\s]*)\s*(?P<url_from_client>[^\s]*)\s*(?P<server_ip>[^\s]*)\s*(?P<user_name>[^\s]*)\s*(?P<hierarchy_status>[^/]*)/(?P<server_peer_name>[^\s]*)\s*(?P<MIME_content_type>[^\s]*)\s*\"(?P<Referer>[^\"]*)\"\s*\"(?P<User_Agent>[^\"]*)\"\s*"""
                     ,line)
        temp_list = []#main_DFに追加する順にparseしたものをいれるリスト
        for feature_name in col_of_DF:#各特徴量をリストに追加
            temp_list.append(r.group(feature_name))#各特徴量をtest_listに追加
        main_DF = main_DF.append(pd.DataFrame([temp_list],columns=col_of_DF,index=[i]))#columnを合わせてDFに追加
        i = i+1#indexを進める

        #if i >10:break#デバッグ用#ループ回数(読み込むログの数)を指定
        #for文終了
    for column in main_DF.columns:main_DF[column] = main_DF[column].astype('category')#main_DFの全ての列を"category"にする("object"だと操作が重くなる)
    joblib.dump(main_DF, dfdir, compress=True)#compressは0~9の値をとり、大きい程圧縮率UP．指定なしなら圧縮されない.TRUEは3と同値．
    return main_DF#出来上がったDataFrameを出力

##Make_PlaneDF_from_log終わり

def Make_clientIP_DF_from_PlaneDF(clientIP,dfdir):
    """
        機能:
            読み込んだPlaneDFファイルからclient_IP属性が入力clientIPに等しい行だけ抽出して新しいDFにする
            ここで作成したDFでセッションごとの移り変わりなどを見る
            保存場所は"./DFs/clientIP_(読み込んだログ名)"
        引数の説明:
            clientIP:
            dfdir:元のDFファイルの名前(正確には./DFs以下のパス)
    """
    PlaneDF = joblib.load("./DFs/"+dfdir)#PlaneDF(ログからDFにparseだけしたもの)を読み込み
    clientDF = PlaneDF[PlaneDF["client_ip"]==clientIP]
    joblib.dump(clientDF,"./DFs/"+clientIP+"_"+dfdir,compress=True)
    return clientDF

##Make_clientIP_DF_from_PlaneDF終わり
    
#serve_IP_binaryが選ばれたときに追加する特徴量のリスト
IP_binary_list = []
for i in range(4):
    for j in range(8):
        IP_binary_list.append("IP_binary_"+str(i)+"_"+str(j))
            
def IP_to_binary(IP):
    """
    機能の説明:
        読み込んだIPアドレスをバイナリ化して32個の文字の配列にして渡す
    引数の説明:
        IP:バイナリ化したいIPアドレスを文字列であらわしたもの 
    例:IP_to_binary("192.168.0.1") →　['1', '1', '0', '0', '0', '0', '0', '0', '1', '0', '1', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1']
    """
    r_IP = re.match(r"""^(?P<IP_0>[0-9]*)\.(?P<IP_1>[0-9]*)\.(?P<IP_2>[0-9]*)\.(?P<IP_3>[0-9]*)"""
                     ,IP)
    return list(format(int(r_IP.group("IP_0")),"08b") + format(int(r_IP.group("IP_1")),"08b") + format(int(r_IP.group("IP_2")),"08b") + format(int(r_IP.group("IP_3")),"08b"))
  
##IP_to_binary終わり
    
def Make_InputDF_from_DF(dfdir,feature_list,num_feature_list,name_support="",rand=False):
    """
        機能の説明:
            読み込んだDFファイル(パースだけ，もしくはclient_ipでの抽出も終えたもの)から特徴量として使いたい特徴量だけを抜き出し、成形して分類器の入力となるDFファイルを出力する
        引数の説明:
            dfdir:元のDFファイルの名前(正確には./DFs以下のパス)
            feature_list:学習に用いる特徴量の名前(下のmatch部分と同じ名前)をリスト形式で指定
            num_feature_list:feature_listの内で数値データのもの,スケーリングをまとめて行うのに用いる
            name_support:作成するDFの名前に任意の文字列を追加する
        特徴量の説明:
            day,month,year,hour,minute,second:ログの時刻
            zero_time:ログを取得したところの基準時刻
            response_time:応答までの時間(ミリ秒)
            client_ip:ユーザのipアドレス
            port:接続ポート
            request_status:リクエストがどうなったか．例:TCP_MISS
            status_to_client:ステータスコード．例:200,403
            size_of_reply:応答のサイズ
###要修正            file_type:URLでアクセスしているファイルの拡張子
            request_method:リクエストされたメソッド．例:GET/POST
            url_from_client:ユーザが指定したurl
            server_ip:上のurlが指すipアドレス
            user_name:ユーザの名前(今のところ空欄が多い)
            hierarchy_status:(理解できなかったので公式の文をそのまま)Squid hierarchy status. 例:DEFAULT_PARENT 
            server_peer_name:通信したサーバの名前
            MIME_content_type:(理解できなかったので公式の文をそのまま)MIME content type
            Referer:リクエストurl
            User_Agent:ユーザの環境(OSやブラウザの情報)
            URL_length:URLの全長
            URL_path_length:URLpathの長さ
            URL_query_length:URLqueryの長さ
            URL_query_key_number:URLqueryに含まれるkeyの数，つまりイコールの数
            URL_have_IP:URLの中にIPアドレスを含んでいるかのフラグ(1ならば含む)
            server_IP_binary:IPアドレスを32桁のバイナリにしてそれぞれ0,1で特徴量として利用
    """
            
    before_DF = joblib.load("./DFs/"+dfdir)#抽出前のDataFrameを読み込み
    
    DF_feature_list = feature_list.copy()#作成するDFのcolumnになる特徴量のリスト,feature_listとは変わる場合があるので別に用意
    
    #server_IP_binaryが選ばれたのでDF_feature_listにIP_binary_listを追加
    if("server_IP_binary" in DF_feature_list):
        DF_feature_list.remove("server_IP_binary")
        DF_feature_list.extend(IP_binary_list)
        
    file_type_list = ["txt","png","jpg","html","htm","ini","gif","crl","php","cab","js","mp4","ico","mp3","zip","css","dat","dll","exe","sys"]#file_typeで認識する拡張子のリスト.ここに入っていないものは"other"に変換する
    file_type_list_columns = []
    for temp_file_type in file_type_list:
        file_type_list_columns.append("file_type_"+temp_file_type)
    file_type_list_columns.append("file_type_other")
    file_type_list_columns.append("file_type_")
    
    #status codeの分類のリストを先につくっておく　(全種類出てこなかった場合に特徴量の数が変わってしまう)
    status_to_client_list=["1","2","3","4","5"]
    status_to_client_list_columns=["status_to_client_1","status_to_client_2","status_to_client_3","status_to_client_4","status_to_client_5"]
    
    after_DF = pd.DataFrame()#学習に使うものだけをいれる空のDataFrameを作成
    
    dic_ip = joblib.load("./Dic/dic_ip.cmp")#ipのブラックリストを読み込み
    dic_url = joblib.load("./Dic/dic_url.cmp")#urlのブラックリストを読み込み
    
    i = 0#デバッグ用
    for line in before_DF.index:#before_DFの各行に対して操作
        url_of_r = parse.urlparse(before_DF.ix[line,"url_from_client"])#ユーザが指定したurlをparseする
        
        if(before_DF.ix[line,"server_ip"]=="-"):
            line_IP_binary=[]
            for i_ip in range(32):line_IP_binary.append("0")#もし接続元IPが不明ならば0.0.0.0としてあつかう
        else:
            line_IP_binary = IP_to_binary(before_DF.ix[line,"server_ip"])#今見ている行のserver_IPをバイナリ化
        
        temp_datas_list = []
        temp_columns_list = []
        for feature in DF_feature_list:#各特徴量をリストに追加
            #以下の２つから小さいDFを作ってtemp_DFにconcatする
            temp_datas = []#temp_DFに追加するデータのリスト
            temp_columns = [feature]#temp_DFに追加するデータのcolumnのリスト
            if(feature == "server_ip"):#ipのブラックリストのdicと比較
                temp = 0.0
                if(before_DF.ix[line,feature] in dic_ip):
                    temp = 1.0
                temp_datas = [temp]
            elif(feature == "server_peer_name"):#urlのブラックリストのdicと比較
                temp = 0.0
                if(before_DF.ix[line,feature] in dic_url):
                    temp = 1.0
                temp_datas = [temp]
            elif(feature == "status_to_client"):#statusコードを大分類に直す(100の桁だけにする)
                temp = int(before_DF.ix[line,feature])
                temp = int((temp - temp%100)/100)
                temp_datas = [0.0,0.0,0.0,0.0,0.0]#まずは全部0
                temp_columns = status_to_client_list_columns.copy()
                temp_datas[temp-1] = 1.0#該当箇所だけフラグを立てる  index番号がずれているので注意
            elif(feature == "size_of_reply"):#返答のサイズを数値に変換
                temp = float(before_DF.ix[line,feature])
                temp_datas = [temp]
            elif(feature == "file_type"):#アクセスしたファイルの拡張子が該当するものであるか
                #アクセスしているのがファイルであればその拡張子をtemp_listに追記
                temp_columns = file_type_list_columns.copy()
                url_path,url_ext = splitext(url_of_r.path)
                url_ext = url_ext.replace(".","")#正常なURLであれば拡張子
                url_ext = url_ext.replace("?","")#パラメタ指定の?が残っていれば除去
                temp=""
                #if(str(url_ext)):#ファイルならばその拡張子を特徴量に設定
                temp="file_type_"+url_ext#特徴量の名前に変換
                if(not (url_ext in file_type_list)):temp="file_type_"+"other"#事前に用意した拡張子のリストに該当するならば
                for temp_i in temp_columns:temp_datas.append(0.0)#0がtemp_columnsの要素数だけ並んリストを作成
                temp_datas[temp_columns.index(temp)]=1.0#該当箇所だけフラグを立てる
#                print(str(line)+":"+temp)#デバッグ用
            elif(feature == "URL_length"):#URLの長さを返す
                temp=float(len(before_DF.ix[line,"url_from_client"]))
                temp_datas = [temp]
            elif(feature == "URL_path_length"):#URLのうちのpathの長さを返す
                temp=float(len(url_of_r[2]))
                temp_datas = [temp]
            elif(feature == "URL_query_length"):#URLのうちのqueryの長さを返す
                temp=float(len(url_of_r[3]))
                temp_datas = [temp]
            elif(feature == "URL_query_key_number"):#URLqueryの中に含まれるイコールの数を返す
                temp=float(url_of_r[3].count("="))
                temp_datas = [temp]
            elif(feature == "URL_have_IP"):#URLの中にIPが含まれているか正規表現で検査
                if(re.search("[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",before_DF.ix[line,"url_from_client"])):
                    temp=1.0
                else:temp=0.0
                temp_datas = [temp]
            elif(feature.find("IP_binary") != -1):#通信先のIPアドレスのバイナリをそれぞれ別々の特徴量にする
                tempi = int(feature[10]);tempj = int(feature[12]);#IPアドレスのどの位置かを名前から抜き出す
                temp=float(line_IP_binary[tempi*8+tempj])
                temp_datas = [temp]
#                print(str(i)+feature+":"+str(temp))
            else:#その他はそのまま特徴量を設定
                temp_datas = [before_DF.ix[line,feature]]
            temp_datas_list.extend(temp_datas)
            temp_columns_list.extend(temp_columns)
        temp_DF = pd.DataFrame([temp_datas_list],columns=temp_columns_list,index=[line])
        after_DF = pd.concat([after_DF,temp_DF],ignore_index=0)#columnを合わせてDFに追加
        i = i+1#デバッグ用
#        print(after_DF.ix[[line],:])#デバッグ用#reでparseしたものを全て表示
#        if i >1:break#デバッグ用#ループ回数(読み込むログの数)を指定
        #for文終了
    #数値データをスケーリング    
    from sklearn.preprocessing import MinMaxScaler
    for num_feature in num_feature_list:
        after_DF[num_feature] = MinMaxScaler().fit_transform(after_DF[num_feature])
    #ダミー変数化
    after_DF = pd.get_dummies(after_DF)
    #randがTrueならば0.0,1.0の2値データに乱数のせて連続数化
    if(rand):
        for category_feature in [c for c in list(after_DF.columns) if c not in num_feature_list]:#非連続特徴量に対して
            after_DF[category_feature] += np.random.rand(len(after_DF.index))#乱数を足すことで0,1の2値グループを維持したまま連続値っぽくする 乱数は大体-0.5~0.5で収まる正規分布
            after_DF[category_feature] += -0.5
    if(name_support):name_support += "_"#補助名が入力されていたら、区切りのアンダーバーを追加
    if(rand):name_support_rand = "_continuous"#2値データを連続数化したら名前に追加
    else:name_suport_rand=""
    joblib.dump(after_DF,"./DFs/Input_DFs/"+"Input_"+name_support+dfdir+name_support_rand,compress = True)#Input_DFsの中に保存
    return after_DF#出来上がったDataFrameをダミー変数化してから出力

##Make_InputDF_from_DF終わり
    
def Extract_from_DF_kmeans(dfdir,num,mode=True):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        modeがFalseのときはシーケンスが既にあっても上書き作成
        
    """
    flag = exists("Database/KMeans/km_full_"+dfdir+"_database_name")#namelistが存在するかどうか
    if(flag and mode):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/KMeans/Result_km_"+str(num)+"_full_Input_"+dfdir+"_continuous")
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    database_name = []
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        result_list = list(result_df.loc[list(plane_df[plane_df["client_ip"]==ip].index)].values.flatten())#client_IPでシーケンス作成
        database.append(result_list)
        database_name.append(ip)
        #if(len(list(set(result_list)))>1):print("        "+ip+"_"+sip+" : "+str(result_list))
    joblib.dump(database,"Database/KMeans/km_"+str(num)+"_full_"+dfdir+"_database")
    if(not(flag)):joblib.dump(database_name,"Database/KMeans/km_full_"+dfdir+"_database_name")
    return [database,database_name]
    
def Extract_from_DF_kmeans_time(dfdir,num,time=3600,mode=True):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        modeがFalseのときはシーケンスが既にあっても上書き作成
        さらに、指定された時間以上の間が開く場合は次のシーケンスにする(標準は1時間以上の開き)
    """
    flag = exists("Database/KMeans/km_full_"+dfdir+"_database_time_name")#namelistが存在するかどうか
    if(flag and mode):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/KMeans/Result_km_"+str(num)+"_full_Input_"+dfdir+"_continuous")
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    database_name = []
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        result_list = list(result_df.loc[list(plane_df[plane_df["client_ip"]==ip].index)].values.flatten())#client_IPでシーケンス作成
        database.append(result_list)
        database_name.append(ip)
        #if(len(list(set(result_list)))>1):print("        "+ip+"_"+sip+" : "+str(result_list))
    joblib.dump(database,"Database/KMeans/km_"+str(num)+"_full_"+dfdir+"_database_time")
    if(not(flag)):joblib.dump(database_name,"Database/KMeans/km_full_"+dfdir+"_database_time_name")
    return [database,database_name]

def Extract_from_DF_kmeans_many(dfdir,num,many=10,mode=True):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        modeがFalseのときはシーケンスが既にあっても上書き作成
        さらに、指定された回数でシーケンスを区切る(標準は10個で区切る)
    """
    flag = exists("Database/KMeans/km_"+str(num)+"_many_is_"+str(many)+"_full_"+dfdir+"_database_many_name")#namelistが存在するかどうか
    if(flag and mode):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/KMeans/Result_km_"+str(num)+"_full_Input_"+dfdir+"_continuous")
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    database_name = []
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        temp_many = 0
        temp_index_list = list(plane_df[plane_df["client_ip"]==ip].index)
        while(len(temp_index_list)>=10):
            database.append(result_df.loc[temp_index_list[0:10]].values.flatten())
            database_name.append(ip+"_"+str(temp_many))
            temp_index_list = temp_index_list[10:]
            temp_many += 1
        if(len(temp_index_list)<=1):continue
        result_list = list(result_df.loc[temp_index_list].values.flatten())#client_IPでシーケンス作成
        database.append(result_list)
        database_name.append(ip+"_"+str(temp_many))
    joblib.dump(database,"Database/KMeans/km_"+str(num)+"_many_is_"+str(many)+"_full_"+dfdir+"_database_many")
    if(not(flag) or not(mode)):
        print("make name files")
        joblib.dump(database_name,"Database/KMeans/km_"+str(num)+"_many_is_"+str(many)+"_full_"+dfdir+"_database_many_name")        
    return [database,database_name]  
    
def Extract_from_DF_kmeans_pair(dfdir,num):
    """
        PlaneDFを読み込んで、(client_IP,server_IP)の組毎に該当index番号の羅列をそれぞれのtxtに書き出す
        命名規則:例:DF_access.log-20171001に含まれる192.168.0.1のindexはIndex_192.168.0.1_DF_access.log-20171001.txt
    """
    if(exists("Database/KMeans/km_full_"+dfdir+"_database_name")):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/KMeans/Result_km_"+str(num)+"_full_Input_"+dfdir)
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    flag = exists("Database/KMeans/km_full_"+dfdir+"_database_name")#namelistが存在するかどうか
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        temp_df = plane_df[plane_df["client_ip"]==ip]#該当client_ipの部分だけ抜き出し
        s_iplist = list(set(temp_df["server_ip"]))#temp_dfに含まれるserver_ipのリスト(重複はsetによって削除済み) 
        for sip in s_iplist:
            result_list = list(result_df.loc[list(temp_df[temp_df["server_ip"]==sip].index)].values.flatten())#1シーケンスに当たる
            database.append(result_list)
            if(not(flag)):database_name.append(ip+"_"+sip)
            if(len(list(set(result_list)))>1):print("        "+ip+"_"+sip+" : "+str(result_list))
    joblib.dump(database,"Database/KMeans/km_"+str(num)+"_full_"+dfdir+"_database")
    if(not(flag)):joblib.dump(database_name,"Database/KMeans/km_full_"+dfdir+"_database_name")
    return [database,database_name]    
        
def Extract_from_DF_kmeans_session(dfdir,num):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        命名規則:例:DF_access.log-20171001に含まれる192.168.0.1のindexはIndex_192.168.0.1_DF_access.log-20171001.txt
    """
    if(exists("Database/KMeans/km_full_"+dfdir+"_database_name")):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/KMeans/Result_km_"+str(num)+"_full_Input_"+dfdir)
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)_(3文字の左0埋め番号)
    for ip in iplist:
        pair_dic={}#組み合わせ毎に出現回数をメモする．それに合わせてシーケンスの名前を作成してdatabase_nameに追加
        server_ip=""
        result_list=[]#シーケンス
        for index in plane_df[plane_df["client_ip"]==ip].index:#client_ip=ipの中で検索
            temp_ip = plane_df.ix[index,"server_ip"]
            if(server_ip==""):server_ip = temp_ip#初期実行の処理
            if(temp_ip == server_ip):
                result_list.append(result_df.ix[index,"label"])
            else:
                sequence_name = ip+"_"+server_ip+"_"
                if(sequence_name in pair_dic):
                    pair_dic[sequence_name] += 1
                else:
                    pair_dic[sequence_name] = 0
#                print(server_ip+" , "+temp_ip)
#                print(sequence_name+" : "+str(result_list)+" , "+str(pair_dic[sequence_name]))
                database.append(result_list)
                database_name.append(sequence_name+"{:0>3}".format(pair_dic[sequence_name]))
                result_list=[result_df.ix[index,"label"]]
                server_ip = temp_ip
#        joblib.dump(list(result_df[plane_df["client_ip"]==ip].values.flatten()),"./List/KMeans/km_"+str(num)+"_full_"+ip+"_"+dfdir)
    joblib.dump(database,"Database/KMeans/km_"+str(num)+"_full_"+dfdir+"_database")
    joblib.dump(database_name,"Database/KMeans/km_"+str(num)+"_full_"+dfdir+"_database_name")
    return [database,database_name]
    
def Extract_from_DF_GMM(dfdir,num,mode=True):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        modeがFalseのときはシーケンスが既にあっても上書き作成
        
    """
    flag = exists("Database/GMM/gmm_full_"+dfdir+"_database_name")#namelistが存在するかどうか
    if(flag and mode):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/GMM/Result_gmm_"+str(num)+"_full_Input_"+dfdir+"_continuous")
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    database_name = []
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        result_list = list(result_df.loc[list(plane_df[plane_df["client_ip"]==ip].index)].values.flatten())#client_IPでシーケンス作成
        database.append(result_list)
        database_name.append(ip)
        #if(len(list(set(result_list)))>1):print("        "+ip+"_"+sip+" : "+str(result_list))
    joblib.dump(database,"Database/GMM/gmm_"+str(num)+"_full_"+dfdir+"_database")
    if(not(flag)):joblib.dump(database_name,"Database/GMM/gmm_full_"+dfdir+"_database_name")
    return [database,database_name]
    
def Extract_from_DF_GMM_time(dfdir,num,time=3600,mode=True):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        modeがFalseのときはシーケンスが既にあっても上書き作成
        さらに、指定された時間以上の間が開く場合は次のシーケンスにする(標準は1時間以上の開き)
    """
    flag = exists("Database/GMM/gmm_full_"+dfdir+"_database_time_name")#namelistが存在するかどうか
    if(flag and mode):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/GMM/Result_gmm_"+str(num)+"_full_Input_"+dfdir+"_continuous")
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        result_list = list(result_df.loc[list(plane_df[plane_df["client_ip"]==ip].index)].values.flatten())#client_IPでシーケンス作成
        database.append(result_list)
        database_name.append(ip)
        #if(len(list(set(result_list)))>1):print("        "+ip+"_"+sip+" : "+str(result_list))
    joblib.dump(database,"Database/GMM/gmm_"+str(num)+"_full_"+dfdir+"_database_time")
    if(not(flag)):joblib.dump(database_name,"Database/GMM/gmm_full_"+dfdir+"_database_time_name")    
    return [database,database_name]
        
def Extract_from_DF_GMM_many(dfdir,num,many=10,mode=True):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        modeがFalseのときはシーケンスが既にあっても上書き作成
        さらに、指定された時間以上の間が開く場合は次のシーケンスにする(標準は1時間以上の開き)
    """
    flag = exists("Database/GMM/gmm_full_"+dfdir+"_database_many_name")#namelistが存在するかどうか
    if(flag and mode):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/GMM/Result_gmm_"+str(num)+"_full_Input_"+dfdir+"_continuous")
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        temp_many = 0
        temp_index_list = list(plane_df[plane_df["client_ip"]==ip].index)
        while(len(temp_index_list)>=10):
            database.append(result_df.loc[temp_index_list[0:10]].values.flatten())
            database_name.append(ip+"_"+str(temp_many))
            temp_index_list = temp_index_list[10:]
            temp_many += 1
        if(len(temp_index_list)<=1):continue
        result_list = list(result_df.loc[temp_index_list].values.flatten())#client_IPでシーケンス作成
        database.append(result_list)
        database_name.append(ip+"_"+str(temp_many))
    joblib.dump(database,"Database/GMM/gmm_"+str(num)+"_full_"+dfdir+"_database_many")
    if(not(flag)):joblib.dump(database_name,"Database/GMM/gmm_full_"+dfdir+"_database_many_name")            
    return [database,database_name]
    
def Extract_from_DF_GMM_pair(dfdir,num):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        命名規則:例:DF_access.log-20171001に含まれる192.168.0.1のindexはIndex_192.168.0.1_DF_access.log-20171001.txt
    """
    if(exists("Database/GMM/gmm_full_"+dfdir+"_database_name")):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/GMM/Result_gmm_"+str(num)+"_full_Input_"+dfdir)
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    flag = exists("Database/GMM/gmm_full_"+dfdir+"_database_name")#namelistが存在するかどうか
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        temp_df = plane_df[plane_df["client_ip"]==ip]#該当client_ipの部分だけ抜き出し
        s_iplist = list(set(temp_df["server_ip"]))#temp_dfに含まれるserver_ipのリスト(重複はsetによって削除済み) 
        for sip in s_iplist:
            result_list = list(result_df.loc[list(temp_df[temp_df["server_ip"]==sip].index)].values.flatten())#シーケンス
            database.append(result_list)
            if(not(flag)):database_name.append(ip+"_"+sip)
            if(len(list(set(result_list)))>1):print("        "+ip+"_"+sip+" : "+str(result_list))
    joblib.dump(database,"Database/GMM/gmm_"+str(num)+"_full_"+dfdir+"_database")
    if(not(flag)):joblib.dump(database_name,"Database/GMM/gmm_full_"+dfdir+"_database_name")
    return [database,database_name]    
        
def Extract_from_DF_dbscan(dfdir,eps,mode=True):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        modeがFalseのときはシーケンスが既にあっても上書き作成
        
    """
    flag = exists("Database/DBSCAN/dbscan_full_"+dfdir+"_database_name")#namelistが存在するかどうか
    if(flag and mode):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/DBSCAN/Result_dbscan_full_"+str(eps)+"_Input_"+dfdir+"_continuous")
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    database_name = []
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        result_list = list(result_df.loc[list(plane_df[plane_df["client_ip"]==ip].index)].values.flatten())#client_IPでシーケンス作成
        database.append(result_list)
        database_name.append(ip)
        #if(len(list(set(result_list)))>1):print("        "+ip+"_"+sip+" : "+str(result_list))
    joblib.dump(database,"Database/DBSCAN/dbscan_"+str(eps)+"_full_"+dfdir+"_database")
    if(not(flag)):joblib.dump(database_name,"Database/DBSCAN/dbscan_full_"+dfdir+"_database_name")
    return [database,database_name]
    
def Extract_from_DF_dbscan_time(dfdir,eps,time=3600,mode=True):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        modeがFalseのときはシーケンスが既にあっても上書き作成
        さらに、指定された時間以上の間が開く場合は次のシーケンスにする(標準は1時間以上の開き)
    """
    flag = exists("Database/DBSCAN/dbscan_full_"+dfdir+"_database_time_name")#namelistが存在するかどうか
    if(flag and mode):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/DBSCAN/Result_dbscan_full_"+str(eps)+"_Input_"+dfdir+"_continuous")
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    database_name = []
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        result_list = list(result_df.loc[list(plane_df[plane_df["client_ip"]==ip].index)].values.flatten())#client_IPでシーケンス作成
        database.append(result_list)
        database_name.append(ip)
        #if(len(list(set(result_list)))>1):print("        "+ip+"_"+sip+" : "+str(result_list))
    joblib.dump(database,"Database/DBSCAN/dbscan_"+str(eps)+"_full_"+dfdir+"_database_time")
    if(not(flag)):joblib.dump(database_name,"Database/DBSCAN/dbscan_full_"+dfdir+"_database_time_name")
    return [database,database_name]
        
def Extract_from_DF_dbscan_many(dfdir,eps,many=10,mode=True):
    """
        PlaneDFを読み込んで、client_IP毎に該当index番号の羅列をそれぞれのtxtに書き出す
        modeがFalseのときはシーケンスが既にあっても上書き作成
        さらに、指定された回数でシーケンスを区切る(標準は10個で区切る)
    """
    flag = exists("Database/DBSCAN/dbscan_full_"+dfdir+"_database_many_name")#namelistが存在するかどうか
    if(flag and mode):return
    plane_df = joblib.load("./DFs/"+dfdir)
    result_df = joblib.load("./DFs/Results/DBSCAN/Result_dbscan_full_"+str(eps)+"_Input_"+dfdir+"_continuous")
    iplist=list(set(plane_df["client_ip"]))#読み込んだDFに含まれるclient_ipのリスト(重複はsetによって削除済み)
    joblib.dump(iplist,"./List/iplist_"+dfdir)#iplistを出力:異常検知に各シーケンスを入れるときに利用
    database = []#シーケンスをどんどん追加して最後に出力する
    database_name = []
    if(not(flag)):database_name = []#シーケンス毎の名前を記録 命名規則:(client_ip)_(server_ip)
    for ip in iplist:
        temp_many = 0
        temp_index_list = list(plane_df[plane_df["client_ip"]==ip].index)
        while(len(temp_index_list)>=10):
            database.append(result_df.loc[temp_index_list[0:10]].values.flatten())
            database_name.append(ip+"_"+str(temp_many))
            temp_index_list = temp_index_list[10:]
            temp_many += 1
        if(len(temp_index_list)<=1):continue
        result_list = list(result_df.loc[temp_index_list].values.flatten())#client_IPでシーケンス作成
        database.append(result_list)
        database_name.append(ip+"_"+str(temp_many))
    joblib.dump(database,"Database/DBSCAN/dbscan_"+str(eps)+"_full_"+dfdir+"_database_many")
    if(not(flag) or not(mode)):
        print("make name files")
        joblib.dump(database_name,"Database/DBSCAN/dbscan_full_"+dfdir+"_database_many_name")        
    return [database,database_name]