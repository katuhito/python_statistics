#推測統計に必要なライブラリ
from os import replace

from matplotlib import colors
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


#いかさまサイコロ
dice = [1,2,3,4,5,6]
prob = [1/21, 2/21, 3/21, 4/21, 5/21, 6/21]
#確率変数の試行にnp.random.choice関数を使う
#引数pにpropを渡すことでそれぞれの確率を指定して、1回試行する
np.random.choice(dice, p=prob)

#上記の試行を100回行う
num_trial = 100
sample = np.random.choice(dice, num_trial, p=prob)
sample

#度数分布表を作成する
freq, _ = np.histogram(sample, bins=6, range=(1, 7))
pd.DataFrame({'度数':freq, '相対度数':freq / num_trial}, index = pd.Index(np.arange(1, 7), name='出目'))

#度数分布によって出目や回数の割合が判明したので、実際の確率分布とともにヒストグラムも図示してみる
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=6, range=(1, 7), density=True, rwidth=0.8)
#真の確率分布を横線で表示
ax.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
#棒グラフの[1.5, 2.5, …, 6.5]の場所に目盛りを付ける
ax.set_xticks(np.linspace(1.5, 6.5, 6))
#目盛りの値は[1,2,3,4,5,6]
ax.set_xticklabels(np.arange(1, 7))
ax.set_xlabel('出目')
ax.set_ylabel('相対度数')
plt.show()

#試行回数を10000回にしたときのヒストグラム
num_trial = 10000
sample = np.random.choice(dice, size=num_trial, p=prob)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=6, range=(1, 7), density=True, rwidth=0.8)
ax.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
ax.set_xticks(np.linspace(1.5, 6.5, 6))
ax.set_xticklabels(np.arange(1, 7))
ax.set_xlabel('出目')
ax.set_ylabel('相対度数')
plt.show()


#推測統計による確率
#階級幅を1点にしてヒストグラムを図示してみる
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(scores, bins=100, range=(0, 100), density=True)
ax.set_xlim(20, 100)
ax.set_ylim(0, 0.042)
ax.set_xlabel('点数')
ax.set_ylabel('相対度数')
plt.show()

#1回試行してみる
np.random.choice(scores)

#無作為抽出においても標本のサンプルサイズを増やしていくと、標本データの相対度数は実際の確率分布に近づく
#無作為抽出によってサンプルサイズ10000の標本を抽出して、その結果をヒストグラムに図示してみる
sample = np.random.choice(scores, 10000)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=100, range=(0, 100), density=True)
ax.set_xlim(20, 100)
ax.set_ylim(0, 0.042)
ax.set_xlabel('点数')
ax.set_ylabel('相対度数')
plt.show()


#標本平均について、標品ひとつひとつが確率変数であるので、それらの平均として計算される標本平均もまた確率変数となる
#無作為抽出により、サンプルサイズ20の標本を抽出して標本平均を計算するという作業を10000回行う
#その結果をヒストグラムに図示することで、標本平均の分布を見る
sample_means = [np.random.choice(scores, 20).mean() for _ in range(10000)]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample_means, bins=100, range=(0, 100), density=True)
#母平均を縦線で表示
ax.vlines(np.mean(scores), 0, 1, 'gray')
ax.set_xlim(50, 90)
ax.set_ylim(0, 0.13)
ax.set_xlabel('点数')
ax.set_ylabel('相対度数')
plt.show()
















