"""代表的な離散型確率分布"""
#推測統計では限られた標本から母集団の平均や分散といった指標を推定することが目的である。しかし、母集団の確率分布の形状に何も置かないでそのような指標を推定することは簡単なものではない。このように母集団の確率分布に何の家庭も置かないことを「ノンパラメトリック」な手法という。
#パラメトリック：ノンパラメトリックと対象となるのがパラメトリックな手法である。これは、母集団はこういう性質のはずだからこんな形状を持った確率分布だろうとある程度仮定を置いて、後は確率分布の期待値や分散を決定する小数のパラメタのみを推測する方法である。

#ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

%precision 3
%matplotlib inline

# グラフの線の種類
linestyles = ['-', '--', ':']

def E(X, g=lambda x: x):
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])

def V(X, g=lambda x: x):
    x_set, f = X
    mean = E(X, g)
    return np.sum([(g(x_k)-mean)**2 * f(x_k) for x_k in x_set])

def check_prob(X):
    x_set, f = X
    prob = np.array([f(x_k) for x_k in x_set])
    assert np.all(prob >= 0), '負の確率があります'
    prob_sum = np.round(np.sum(prob), 6)
    assert prob_sum == 1, f'確率の和が{prob_sum}になりました'
    print(f'期待値は{E(X):.4}')
    print(f'分散は{(V(X)):.4}')

def plot_prob(X):
    x_set, f = X
    prob = np.array([f(x_k) for x_k in x_set])

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(111)
    ax.bar(x_set, prob, label='prob')
    ax.vlines(E(X), 0, 1, label='mean')
    ax.set_xticks(np.append(x_set, E(X)))
    ax.set_ylim(0, prob.max()*1.2)
    ax.legend()

    plt.show()


"""ベルヌーイ分布"""
#ベルヌーイ分布は、最も基本的な離散型確率分布で、取り得る値が0と1しかない確率分布である。ベルヌーイ分布に従う確率変数の試行のことをベルヌーイ試行といい、1が出ることを成功、0が出ることを失敗という。
#取り得る値が2つしかなく、確率の和が1という性質から、どちらかの確率が定まればもう一方の確率が自動的に定まる。そのためベルヌーイ分布では1が出る確率をp,0が出る確率を1-pとする。このpがベルヌーイ分布の形を調整できる唯一のパラメタで、確率の性質を満たすために0<p<1を満たす必要がある

#ベルヌーイ分布の期待値と分散
#ベルヌーイ分布をNumPyで実装してみる。パラメタを定めることにより確率分布が確定するので、パラメタpを引数にx_setとｆを返す関数で実装する。
    
def Bren(p):
    x_set = np.array([0,1])
    def f(x):
        if x in x_set:
            return p ** x * (1-p) ** (1-x)
        else:
            return 0
    return x_set, f

#Bern(0.3)に従う確率変数Xをつくってみる。
p = 0.3
X = Bren(p)

#期待値と分散を計算
check_prob(X)

#確率変数Xを図示する。中央の縦線が確率変数Xの期待値を示している
plot_prob(X)

#scipy.statsを使った実装
#scipy.statsにはベルヌーイ分布にしたがy確率関数を造ることができるbernoulli関数がある。ベルヌーイ関数は引数にパラメタpをとり、返り値としてbern(p)に従うrv_frozen objectを返す。
#rv_frozen objectはscipy.statusにおける確率変数に相当するもので、様々なメソッドを持っている。
rv = stats.bernoulli(p)

#rvのpmfメソッドは確率関数を計算できる。0と1をそれぞれ渡すと、その値を取る確率が返ってくる。
rv.pmf(0), rv.pmf(1)

#pmfメソッドは引数リストを渡すこともできる。この場合、リストの各要素に対する確率を格納したNumpyのarrayが返ってくる。
rv.pmf([0,1])

#cdfメソッドを使うことで累積密度関数を計算できる。こちらも引数にリストを渡すことができる。
rv.cdf([0,1])

#meanメソッドやvarメソッドを呼び出すことで期待値や分散を計算できる
rv.mean(), rv.var()


"""二項分布"""
#二項分布は成功確率がpのベルヌーイ試行をn回行ったときの成功回数が従う分布である。成功する回数は0回からn回まであるので、取り得る値は{0,1,…,n}である。
#二項分布のパラメタには成功確率のpと試行回数のnの2つがあり、pは0<p<1で、んは1以上の整数である必要がある。パラメタがn,pの二項分布をBin(n,p)と表記する。
#Bin(n,p) => 二項分布の確率関数 => https://ja.wikipedia.org/wiki/%E4%BA%8C%E9%A0%85%E5%88%86%E5%B8%83

#二項分布の具体例
#10回コインを投げて表が出る回数
#これはp=1/2のベルヌーイ試行を10回行ったときの成功回数と考える事ができるのでBin(10, 1/2)に従う。このことからコインを10回投げて表が3回出る確率は、P(x=3) = 15/128 となる。
#4回サイコロを投げて6が出る回数
#これはp=1/6のベルヌーイ試行を4回行ったときの成功回数と考える事ができるので、Bin(4, 1/6)に従う。このことから4回サイコロを投げて6が1回も出ない確率であれば、　p(x=0) = 625/1296 となる。

#二項分布の期待値と分散
#X ~ Bin(n, p)とするとき、 期待値：E(X)= np,　分散V(X) = np(1-p)

#二項分布をNumpyで実装してみる。コンビネーションnCxの計算には、scipy.specialにあるcomb関数を用いる。
from scipy.special import comb

def Bin(n, p):
    x_set = np.arange(n+1)
    def f(x):
        if x in x_set:
            return comb(n, x) * p**x * (1-p)**(n-x)
        else:
            return 0
    return x_set, f

#Bin(10, 0.3)に従う確率変数Xを作ってみる。
n = 10
p = 0.3
X = Bin(n, p)

#期待値は10*0.3 = 3 分散は10*0.3*0.7 = 2.1となる
check_prob(X)

#図示する。二項分布は期待値でピークをとる山形の分布となる。
plot_prob(X)

#scipy.statsでは二項分布の確率変数はbinom関数によって作ることができる。nを10に固定して、pを0.3,0.5,0,7させて二項分布がどのような形になるか見てみる。
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

x_set = np.arange(n+1)
for p, ls in zip([0.3, 0.5, 0.7], linestyles):
    rv = stats.binom(n, p)
    ax.plot(x_set, rv.pmf(x_set), label=f'p:{p}', ls=ls, color='gray')
ax.set_xticks(x_set)
ax.legend()
plt.show()

 
"""幾何分布"""
#幾何分布はベルヌーイ試行を繰り返して、はじめて成功するまでの試行回数が従う確率分布である。
#幾何分布は1回目で成功することもあれば延々と守パイを続けることもあり得るので、取り得る値は1以上の整数全体{1,2,…}となる。
#幾何分布にパラメタはベルヌーイ試行の成功確率パラメタpとなる。ベルヌーイ試行の成功確率尾パラメタなのでpは0＜ｐ＜1を満たす必要がある。パラメタ幾何分布をGe(p)と表記する。

#幾何分布の確率関数：f(x)=(1-p)**(x-1)*p (x ∋ {1,2,3,…})　f(x)=0

#幾何分布の期待値と分散：X ~ Ge(p)とするとき(期待値：1回の試行で得られる値の平均値、得られる全ての値とそれが起こる確率の積を足し合わせたもの)
                        #期待値：E(X)=1/p　分散：V(X)=(1-p)/p**2

#幾何分布をNumpyで実装する。幾何分布の取り得る値は1以上の整数全てであるが、実装上の都合でここではx_setを1以上29以下の整数としている。

def Ge(p):
    x_set = np.arange(1, 30)
    def f(x):
        if x in x_set:
            return p * (1-p) ** (x-1)
        else:
            return 0
    return x_set, f

#ここでは確率変数XはGe(1/2)に従うとする
p = 0.5
X = Ge(p)

#期待値と分散
check_prob(X)

#図示する。空いたが大きくなるにつれて確率は指数的に減っていき、11以上の値を取る確率はほぼ0になり、グラフからは確認できなくなる。
plot_prob(X)

#scipy.statsでは幾何分布はgeom関数で作ることができる。パラメタpが0.2,0.5.0.8のときの幾何分布図示する。ここではx_setを1以上14以下の整数に設定している。
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

x_set = np.arange(1, 15)
for p, ls in zip([0.2,0.5,0.8], linestyles):
    rv = stats.geom(p)
    ax.plot(x_set, rv.pmf(x_set), label=f'p:{p}', ls=ls, color='gray')
ax.set_xticks(x_set)
ax.legend()

plt.show()


"""ポアソン分布"""
#ポアソン分布は、ランダムな事象が単位時間当たりに発生する件数が従う確率分布である。発生する件数の確率分布なので、取り得る値は{0,1,2,…}となる。そしてポアソン分布のパラメタはλで、λは正の実数である必要がある。
#ポアソン分布Poi(λ)は単位時間当たり平均λ回起こるようなランダムな事象が、単位時間に起こる件数が従う確率分布なので、具体例として次のようなものがある。
    #1日当たり平均2件の交通事故が発生する地域における、1日の交通事故の発生件数
    #=>交通事故を完全にランダムな事象と捉えると、単位時間（一日）あたり発生する交通事故の発生件数は、Poi(2)に従う。
    #1時間当たり平均10アクセスあるサイトへの、1時間当たりのアクセス件数
    #=>サイトへのアクセスを完全にランダムな事象と捉えると、単位時間（1時間）当たりのサイトへのアクセス件数はPoi(λ)に従う。
#ポアソン分布の期待値と分散はどちらもλになる。期待値と分散が同じになるというのはポアソン分布の特徴のひとつである。
    #期待値：E(X)=λ　　　分散：V(X)=λ

#ポアソン分布をNumPyで実装する。階乗x!はscipy.specialのfactorialを使用する。取り得る値は0以上の整数すべてであるが、実装上の都合でx_setを0以上19以下の整数としている。
from scipy.special import factorial

def Poi(lam):
    x_set = np.arange(20)
    def f(x):
        if x in x_set:
            return np.power(lam, x) / factorial(x) * np.exp(-lam)
        else:
            return 0
    return x_set, f

#ここで確率変数XはPoi(3)に従うとする
lam = 3
X = Poi(lam)

#期待値と分散はともに3となる。
check_prob(X)

#図示
plot_prob(X)

#scipy.statusではポアソン分布はpoisson関数で作ることができる。パラメタλを3,5,8で変化させたときの、ポアソン分布を図示する。
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

x_set = np.arange(20)
for lam, ls in zip([3,5,8], linestyles):
    rv = stats.poisson(lam)
    ax.plot(x_set, rv.pmf(x_set), label=f'lam:{lam}', ls=ls, color='gray')
ax.set_xticks(x_set)
ax.legend()

plt.show()






