#推測統計に必要なライブラリ
from os import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%precision 3
%matplotlib inline

#データCSVの読み込み
df = pd.read_csv('./data/ch4_scores400.csv')
scores = np.array(df['点数'])
scores[:10]

#無作為抽出(ランダム性を伴う抽出)
np.random.choice([1, 2, 3], 3)

#非復元抽出(復元抽出)
np.random.choice([1, 2, 3], 3, replace=False)

#乱数シード(これから発生させる乱数の元となる数字を設定する)
np.random.seed(0)
np.random.choice([1, 2, 3], 3)

#Aさんの行った無作為抽出はnp.random.seed(0)で乱数シードを0に指定した後に、scoresから
#サンプルサイズを20で復元抽出することで再現することができる。
#無作為抽出を行い、標本平均を計算する
np.random.seed(0)
sample = np.random.choice(scores, 20)
sample.mean()

#全生徒のデータから、Aさんが推測したい母平均も計算できる
scores.mean()

#無作為抽出は行うたびに結果が異なるため、得られる標本平均も毎回異なる。無作為抽出とその標本平均の計算を何回か行ってみる
for i in range(7):
    sample = np.random.choice(scores, 20)
    print(f'{i+1}回目の無作為抽出で得た標本平均', sample.mean())






