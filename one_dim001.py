from os import name
import numpy as np
import pandas as pd

#jupyter notebookの出力を小数点以下３桁に抑える
%precision 3
#DataFrameの出力を小数点以下３桁に抑える
pd.set_option('precision', 3)

df = pd.read_csv('./data/ch2_scores_em.csv', index_col='生徒番号')
#dfの最初の５桁を標示
df.head()

#英語の点数のみをarray配列へ
scores = np.array(df['英語'])[:10]
scores

#DataFrame=>A~Jの１０人分の点数を表示
scores_df = pd.DataFrame({'点数':scores}, index=pd.Index(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], name='生徒'))
scores_df

#平均値
sum(scores) / len(scores)

#平均値その２
scores_df.mean()

#中央値(median)=>点数を大きさの順番に並べ替える
sorted_scores = np.sort(scores)
sorted_scores

#中央値の定義をコードへ落とす
n = len(sorted_scores)
if n % 2 == 0:
    m0 = sorted_scores[n//2 - 1]
    m1 = sorted_scores[n//2]
    median = (m0 + m1) / 2
else:
    median = sorted_scores[(n+1)//2 - 1]
median

#NumPyのmedian関数
np.median(scores)

#DataFrameやSeriesのmedianメソッド
scores_df.median()

#最頻値(mode)
pd.Series([1, 1, 1, 2, 2, 3]).mode()
pd.Series([1, 2, 3, 4, 5]).mode()


#偏差(deviation)
mean = np.mean(scores)
deviation = scores - mean
deviation

#点数と偏差(偏差の平均は常に０)
summary_df = scores_df.copy()
summary_df['偏差'] = deviation
summary_df
summary_df.mean()

#分散(=偏差の二乗)=>variance
np.mean(deviation ** 2)

#分散(NumPy)
np.var(scores)

#分散(DataFrame/Series)
scores_df.var()

#summary_dfに偏差二乗の列を追加
summary_df['偏差二乗'] = np.square(deviation)
summary_df
summary_df.mean()

#分散の平方根(NumPy)
np.sqrt(np.var(scores, ddof=0))

#分散の平方根(DataFrame/Series)
np.std(scores, ddof=0)

#範囲(range)
np.max(scores) - np.min(scores)

#四分位範囲(interquatile range)
scores_Q1 = np.percentile(scores, 25)
scores_Q3 = np.percentile(scores, 75)
scores_IQR = scores_Q3 - scores_Q1
scores_IQR

#describe =>DataframeやSeriesにはdescribeという様々な指標を一度に求めることができるメソッド=>与えられたデータに対して大まかに概要を得るのに便利である。
pd.Series(scores).describe()


#正規化(normalization),標準化(standardization)=>テストの点数を標準化
z = (scores - np.mean(scores)) / np.std(scores)
z

#標準化されたデータの平均と標準偏差
np.mean(z), np.std(z, ddof=0)

#各生徒の偏差値
z = 50 + 10 * (scores - np.mean(scores)) / np.std(scores)
z

# 点数と偏差値の関係=>Dataframe
scores_df['偏差値'] = z
scores_df
















