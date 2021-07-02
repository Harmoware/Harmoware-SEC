#HTTP requestのみのDFを作成中 他にHTTP responceのみのDFを作ってackで関連を見る
import dpkt
import pandas as pd
import re
import datetime
import socket
from dpkt.compat import compat_ord


def mac_addr(address):
    """Convert a MAC address to a readable/printable string

       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return(':'.join('%02x' % compat_ord(b) for b in address))



def inet_to_str(inet):
    """Convert inet object to a string

        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return(socket.inet_ntop(socket.AF_INET, inet))
    except ValueError:
        return(socket.inet_ntop(socket.AF_INET6, inet))


def Transform_pcap(pcap_file="./D3M_2015/20150208/20150208_marionette.pcap"):
    pcap  = dpkt.pcap.Reader(open(pcap_file,"rb"))

    pd.set_option("display.max_colwidth", 8000)
    pd.set_option("display.max_columns", 400)
    pd.set_option("display.max_rows", 100)
    ind=0#indexナンバーWirreSharkと見比べたりするのに使用
    sq_dic={}#パケットからsquid風ログの作成に使うものを保存していく(timestampも一緒に)
    pre_sequence_dic={}#現在探しているシーケンス番号を指定したパケットのシーケンス番号
    for timestamp, buf in pcap:
        ind += 1
        # Unpack the Ethernet frame (mac src/dst, ethertype)
        eth = dpkt.ethernet.Ethernet(buf)#Ethternetのヘッダとその中身に分ける
        
        # Ethernet形式のdataがIP packetを含場合のみ扱う
        if not isinstance(eth.data, dpkt.ip.IP):
            #print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
            continue

        # IP packetはEthernetのdata部分
        ip = eth.data

        # トランスポート層のプロトコルがTCPである場合のみ扱う
        if isinstance(ip.data, dpkt.tcp.TCP):

            # Set the TCP data
            tcp = ip.data
            #if(ind in [1306,1310,1311,1312,1317,1318,1328,1331,1332,1333,1334,1462,1463,1464,1465,1466,1470,1471,1472]):print(ind,tcp.seq,tcp.ack,len(tcp.data),tcp.flags);
            #if("data=" in repr(tcp)):print("index : ",ind,"    ",repr(tcp))
            #ここでレスポンスの処理を行う リクエストの次のackをもつ
            if(tcp.seq in sq_dic.keys()):
                if(sq_dic[tcp.seq]["data"] is None):#該当セッションのレスポンスの頭がまだ見つかっていない場合
                    if(repr(tcp.data)[0:7] not in["b\"HTTP/","b\'HTTP/"]):#HTTPレスポンスではない場合は探すシーケンス番号をtcp.ack変えて進む
                        if(sq_dic[tcp.seq]["pre_seq"] in pre_sequence_dic.keys()):del(pre_sequence_dic[sq_dic[tcp.seq]["pre_seq"]])#直前に指定していた番号を削除
                        sq_dic[tcp.seq]["pre_seq"] = tcp.seq#今のシーケンス番号を直前のシーケンス番号として登録
                        sq_dic[tcp.ack]=sq_dic[tcp.seq]#次のシーケンス番号待ちに移動
                        del(sq_dic[tcp.seq])
                        pre_sequence_dic[tcp.seq]=tcp.ack#次のシーケンスを指定した現在のシーケンス番号を登録
                        continue
                    sq_dic[tcp.seq]["found_response"]=True#レスポンスを発見したので変更
                    header = repr(tcp.data).split("\\r\\n\\r\\n")[0]#ヘッダとボディで分けてヘッダだけ抜き出し
                    sq_dic[tcp.seq]["status_to_client"] = header.split(" ")[1]#ステータスコードを保存 例:200
#現在はHTTPヘッダも含めてサイズ計算をするのでここはコメントアウト                    sq_dic[tcp.seq]["data"] = re.search(b"\\r\\n\\r\\n(?P<body>[\s\S]*)",tcp.data)["body"]#HTTPリクエストの最初はヘッダを含むのでそこ以外をsq_dic[tcp.seq]["data"]として追加
                    sq_dic[tcp.seq]["data"] = tcp.data#TCPパケットのペイロードを得るためにsq_dic[tcp.seq]["data"]として追加
                    #print(sq_dic[tcp.seq]["index"]," : ",tcp.seq)#," : ",sq_dic[tcp.seq]["size_of_reply"],"\n    ",sq_dic[tcp.seq]["data"])#Content-Lengthが記載されていない場合はそれを表示
                    sq_dic[tcp.seq]["Chunked"] = b"Transfer-Encoding: chunked" in tcp.data
                else:#レスポンスの続きパケットだった場合
                    sq_dic[tcp.seq]["data"] += tcp.data#パケットの全体がbodyなのでdataの後ろに連結
                if(len(tcp.data)==0):continue;#tcp.dataが0になったら終わり
                sq_dic[tcp.seq]["index"].append(ind)
                sq_dic[tcp.seq]["timestamp"].append(timestamp)#関連パケットのtimestampを追加していく
                sq_dic[tcp.seq+len(tcp.data)]=sq_dic[tcp.seq]#レスポンスの続きのシーケンス番号に移動
                if(sq_dic[tcp.seq]["pre_seq"] in pre_sequence_dic.keys()):del(pre_sequence_dic[sq_dic[tcp.seq]["pre_seq"]])
                del(sq_dic[tcp.seq])#移動元の要素を削除
                pre_sequence_dic[tcp.seq]  = tcp.seq+len(tcp.data)
                continue#レスポンスの処理終了            
            elif(tcp.seq in pre_sequence_dic.keys() and tcp.ack == pre_sequence_dic[tcp.seq]):#今のパケットのシーケンス番号が前のパケットと同じかつ、ackが等しい場合
                if(repr(tcp.data)[0:7] in["b\"HTTP/","b\'HTTP/"]):#HTTPレスポンスだった場合は探すシーケンス番号を更新して進む
                    sq_dic[tcp.ack]["timestamp"].append(timestamp)#関連パケットのtimestampを追加していく
                    sq_dic[tcp.ack]["index"].append(ind)
                    sq_dic[tcp.ack]["found_response"] = True#レスポンスを発見
                    header = repr(tcp.data).split("\\r\\n\\r\\n")[0]#ヘッダとボディで分けてヘッダだけ抜き出し
                    sq_dic[tcp.ack]["status_to_client"] = header.split(" ")[1]#ステータスコードを保存 例:200
                    sq_dic[tcp.ack]["data"] = tcp.data#TCPパケットのペイロードを得るためにsq_dic[tcp.seq]["data"]として追加
                    #print(sq_dic[tcp.ack]["index"]," : ",tcp.seq)#," : ",sq_dic[tcp.seq]["size_of_reply"],"\n    ",sq_dic[tcp.seq]["data"])#Content-Lengthが記載されていない場合はそれを表示
                    sq_dic[tcp.ack]["Chunked"] = b"Transfer-Encoding: chunked" in tcp.data#chunkedかのチェック
                del(pre_sequence_dic[tcp.seq])#直前に指定していた番号を削除
                sq_dic[tcp.ack]["pre_seq"] = tcp.ack#今のシーケンス番号を直前のシーケンス番号として登録
                sq_dic[tcp.seq+len(tcp.data)]=sq_dic[tcp.ack]#次のシーケンス番号に移動
                pre_sequence_dic[tcp.ack]=tcp.seq+len(tcp.data)#次のシーケンスを指定した現在のシーケンス番号を登録
                del(sq_dic[tcp.ack])#移動元の要素を削除
                continue#レスポンスの処理終了 
                
            # HTTPのリクエストのみ抜き出していく
            try:
                request = dpkt.http.Request(tcp.data)#リクエスト形式だった場合
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):#その他の場合
                continue#レスポンスでもリクエストでもないものはスルー

            #リクエスト
            if("referer"not in request.headers.keys()):request.headers["referer"]="-"#refererの情報が無かったら"-"で置き換え
            #reqから抽出できるものをsq_dicに突っ込む
#            print("index : ",ind,"    ack : ",tcp.ack,"    next ack : ",tcp.seq+len(tcp.data))#ackの確認
            if("host"not in request.headers.keys()):request.headers["host"]=""#hostを入れずにGetを投げる例があったのでエラーの回避
            sq_dic[tcp.ack] = {"found_response":False,"timestamp":[timestamp],"response_time":None,"client_ip":inet_to_str(ip.src),
                                             "port":tcp.sport,"request_status":None,"status_to_client":None,"size_of_reply":None,
                                             "request_method":request.method,"url_from_client":"http://"+request.headers["host"]+request.uri,
                                             "server_ip":inet_to_str(ip.dst),"user_name":None,"hierarchy_status":None,
                                             "server_peer_name":request.headers["host"],"Referer":request.headers["referer"],
                                             "User_Agent":request.headers["user-agent"],"data":None,"index":[ind],"Chunked":False,"pre_seq":None}
                #レスポンスのシーケンス番号を鍵として(timestamp,ステータスコード,content_length)を保存,
                #found_responseで対応するレスポンスを発見したか判断

    #print(sq_dic.keys())
    #データの連結が終わったのでそのサイズを測る
    for key in sq_dic.keys():
        if(not sq_dic[key]["found_response"]):continue#レスポンスを見つけられなかった場合はとばす
        #if(sq_dic[key]["size_of_reply"]is None):print(sq_dic[key]["index"]," : ",key," : ",sq_dic[key]["size_of_reply"],"\n    ")
        if(len(sq_dic[key]["index"])==1):print(sq_dic[key])
        sq_dic[key]["size_of_reply"]=len(sq_dic[key]["data"])
        sq_dic[key]["response_time"] = int((sq_dic[key]["timestamp"][-1]-sq_dic[key]["timestamp"][0])*1000)
        sq_dic[key]["timestamp"]=[sq_dic[key]["timestamp"][0],sq_dic[key]["timestamp"][-1]]
        #if(sq_dic[key]["size_of_reply"] is None):print(sq_dic[key])
    squid_DF = pd.DataFrame([list(sq_dic[key].values()) for key in list(sq_dic.keys())],columns=list(sq_dic[key].keys()))
    return squid_DF#[squid_DF["Chunked"]==True]