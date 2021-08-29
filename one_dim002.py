"""一次元データの視覚化"""

import numpy as np
import pandas as pd

#５０人の英語の点数のarray
english_scores = np.array(df['英語'])
#Seriesに変換して、describeを表示
pd.Series(english_scores).describe()

#度数分布表を作成する
freq, _ = np.histogram(english_scores, bins=10, range=(0, 100))
freq

#0~10, 10~20, ...といった文字列のリストを作る
freq_class = [f'{i}~{i+10}' for i in range(0, 100, 10)]
#freq_classをインデックスにしてfreqでDataFrameを作る
freq_dist_df = pd.DataFrame({'度数':freq}, index=pd.Index(freq_class, name='階級'))
freq_dist_df

#階級値(階級を代表する値のことで、階級の中央値が使われる)
class_value = [(i+(i+10))//2 for i in range(0, 100, 10)]
class_value

#相対度数は全データに対してその階級のデータがどのくらいの割合で占めているかを示す
rel_freq = freq / freq.sum()
rel_freq

#累積相対度数はその階級までの相対度数の和を示す。
cum_rel_freq = np.cumsum(rel_freq)
cum_rel_freq

#下級値と相対度数と累積相対度数を度数分布表に付け加える
freq_dist_df['階級値'] = class_value
freq_dist_df['相対度数'] = rel_freq
freq_dist_df['累積相対度数'] = cum_rel_freq
freq_dist_df = freq_dist_df[['階級値', '度数', '相対度数', '累積相対度数']]
freq_dist_df

#最頻値再び
#量的データに対しても自然に最頻値を求めることができる
#度数分布表を使った最頻値は度数が最大である階級の階級値で定義される
freq_dist_df.loc[freq_dist_df['度数'].idxmax(), '階級値']

#ヒストグラム
#データ分布の形状を視覚化する=>matplotlib
import matplotlib.pyplot as plt
#グラフがnotebook上に表示されるようにする
%matplotlib inline

#キャンパスを作る
#figsizeで横、縦の大きさを指定
fig = plt.figure(figsize=(10, 6))
#キャンパス上にグラフを描画するための領域を作る
#引数は領域を１×１個作り、１つめの領域に描画することを意味する
ax = fig.add_subplot(111)

#階級数を１０にしてヒストグラムを描画
freq, _, _ = ax.hist(english_scores, bins=10, range=(0, 100))
#X軸にラベルを付ける
ax.set_xlabel('点数')
#y軸にラベルを付ける
ax.set_ylabel('人数')
#X軸に0,10,20,...,100の目盛りをふる
ax.set_xticks(np.linspace(0, 100, 10+1))
#Y軸に0,1,2,...の目盛りをふる
ax.set_yticks(np.arange(0, freq.max()+1))
#グラフの表示
plt.show()


#階級数を25に細分化して、階級幅を４点にしたヒストグラム
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

freq, _ , _ = ax.hist(english_scores, bins=25, range=(0, 100))
ax.set_xlabel('点数')
ax.set_ylabel('人数')
ax.set_xticks(np.linspace(0, 100, 25+1))
ax.set_yticks(np.arange(0, freq.max()+1))
plt.show()

#相対度数のヒストグラムを累積相対度数の折れ線グラフと一緒に描画する
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111)
#Y軸のスケールが違うグラフをax1と同じ領域上に書けるようにする
ax2 = ax1.twinx()

#相対度数のヒストグラムにするためには、度数をデータの数で割る必要がある
#これはhistの引数weightを指定することで実現できる
weights = np.ones_like(english_scores) / len(english_scores)
rel_freq, _, _ = ax1.hist(english_scores, bins=25, range=(0, 100), weights=weights)
cum_rel_freq = np.cumsum(rel_freq)
class_value = [(i+(i+4))//2 for i in range(0, 100, 4)]
#折れ線グラフの描画
#引数lsを'--'にすることでデータ線が点線に
#引数markerを'o'にすることでデータ点を丸に
#引数colorを'gray'にすることで灰色に
ax2.plot(class_value, cum_rel_freq, ls='--', marker='o', color='gray')
#折れ線グラフの罫線を消去
ax2.grid(visible=False)

ax1.set_xlabel('点数')
ax1.set_ylabel('相対度数')
ax2.set_xlabel('累積相対度数')
ax1.set_xticks(np.linspace(0, 100, 25+1))

plt.show()


#箱ひげ図
fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(111)
ax.boxplot(english_scores, labels=['英語'])

plt.show()






