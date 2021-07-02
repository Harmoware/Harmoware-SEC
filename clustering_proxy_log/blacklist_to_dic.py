import re#正規表現のライブラリ
from sklearn.externals import joblib#オブジェクトの保存・読み込みに利用

#matchした部分全部を表示
def re_display(r):
    """
        入力:r:正規表現
        出力:rの中のマッチした部分を全列挙
    """
    for (key,para) in zip(tuple(r.groupdict().keys()),r.groups()):#matchの中で名付けたパートの羅列
        print("%s = %s" % (key,para),end="       ")
    print()



def Make_Dic(file_name,dic_name):
    """
        入力の説明
            file_name:辞書を作りたいブラックリストの相対パス 例:"./Blacklists/bl_url_file.txt"
            dic_name:作った辞書の保存名を相対パスで 例:"./Dic/dic_url.cmp"
    """
    f = open(file_name)
    dic = {}#作成する辞書．最初は空箱
    for line in f:
        #下の行でマッチのパターンを定義
        #row文字列を複数行で書く場合は 1.途中でインデントをいれないこと 2.改行したときにバックスラッシュを入れる
        r = re.match(r"""(?P<ip>.*),(?P<label>[^\s]*)\s*""",line)
        dic[r.group("ip")]=r.group("label")
    joblib.dump(dic, dic_name, compress=True)#compressは0~9の値をとり、大きい程圧縮率UP．指定なしなら圧縮されない.TRUEは3と同値．
