#ライブラリとデータを準備する
from os import name

from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd

%precision 3
pd.set_option('precision', 3)

#データの読み込み
df = pd.read_csv('./data/ch2_scores_em.csv', index_col='生徒番号')

#最初の10人分のデータを用意する
# NumpyのarreyとPandasのDataFrameの両方を用意して、DataFrameには各生徒にA,B...と
# 名付けておく
en_scores = np.array(df['英語'])[:10]
ma_scores = np.array(df['数学'])[:10]
scores_df = pd.DataFrame({'英語':en_scores, '数学':ma_scores}, index=pd.Index(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], name='生徒'))
scores_df

#共分散：(p45)点数の散布と符号付け面積の図において、共分散は正であれば正の面積となるデータが多いということで正の相関を、
#負であれば負の面積になるデータが多いということで負の相関があるといえる。
#正と負どちらでもなく、共分散が0に近ければ無相関を表す。
#DataFreameでまとめつつ、共分散を計算する。
summary_df = scores_df.copy()
summary_df['英語の偏差'] = summary_df['英語'] - summary_df['英語'].mean()
summary_df['数学の偏差'] = summary_df['数学'] - summary_df['数学'].mean()
summary_df['偏差同士の積'] = summary_df['英語の偏差'] * summary_df['数学の偏差']
summary_df

#偏差同士の積の平均値を求める
summary_df['偏差同士の積'].mean()

#NumPyの場合、共分散はcov関数で求めることができる。但し返り値は共分散という値ではなく、共分散行列または、分散共分散行列と呼ばれる行列である。
cov_mat = np.cov(en_scores, ma_scores, ddof=0)
cov_mat

#cov関数で求めた共分散の行列において、1行1列目が第1引数の英語、2行2列目が第2引数の数学にそれぞれ対応しており、それらが交わる1行2列目と2行1列目の成分が英語と数学の共分散(偏差同士の積の平均値)に該当する。
#Pythonのインデックスは0始まりなので、結局cov_matの[0,1]成分と[1,0]成分が共分散である。
cov_mat[0,1], cov_mat[1,0]

#同じ変数同士の共分散はその変数の分散と等しくなっている。[0,0]成分は英語の分散、[1,1]成分は数学の分散になる。
cov_mat[0,0], cov_mat[1,1]

#英語と数学の分散値は分散と一致
np.var(en_scores, ddof=0), np.var(ma_scores, ddof=0)


#相関係数(-1~1)
#英語と数学の相関係数を求めてみる
np.cov(en_scores, ma_scores, ddof=0)[0,1] / (np.std(en_scores) * np.std(ma_scores))

#相関係数をcorrcoef関数で計算すると、返り値は共分散の時と同じ相関行列(correlation matrix)
np.corrcoef(en_scores, ma_scores)

#同様の結果を,DataFrameのcorrメソッドを使って得る
scores_df.corr()












 


