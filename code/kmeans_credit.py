from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pd.set_option('display.max_columns', None)
"""
基于Python的信用评分模型开发.
信用评分模型可用“四张卡”来表示，分别是 A卡（Application score card，申请评分卡）
B卡（Behavior score card，行为评分卡）
C卡（Collection score card，催收评分卡）
F卡（Anti-Fraud Card，反欺诈评分卡），分别应用于贷前、贷中、贷后。

"""


def load_data(path):
    """
    读取数据
    :param path:
    :return:
    """
    data = pd.read_csv(path, encoding='utf8', )
    data.drop(columns='id', inplace=True)
    logging.info(f'前五条数据 \n {data.head()}')
    logging.info(f'NA值的个数\n {data.isnull().sum()}')
    logging.info(f'数据的描述信息\n {data.describe()}')
    logging.info(f'数据的列索引\n {data.columns}')
    logging.info(f'数据的整体信息\n {data.info()}')
    return data


def set_missing(data):  # 缺失值处理
    # １．直接删除含有缺失值的样本。
    # ２．根据样本之间的相似性填补缺失值。
    # ３．根据变量之间的相关关系填补缺失值。
    # MonthlyIncome 缺失率比较大变量之间的相关关系填补缺失值
    # 用随机森林对缺失值预测填充函数
    # 把已有的数值型特征取出来
    process_df = data.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
    # 分成已知该特征和未知该特征两部分
    known = process_df.loc[process_df['MonthlyIncome'].notnull()].values
    unknown = process_df.loc[process_df['MonthlyIncome'].isnull()].values
    # x 为特征属性值
    X = known[:, 1:]
    # y 为结果标签值
    y = known[:, 0]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    logging.info(f'预测结果{predicted}')
    # 用得到的预测结果填补原缺失数据
    data.loc[data['MonthlyIncome'].isnull(), 'MonthlyIncome'] = predicted
    # NumberOfDependents  变量缺失值比较少，可以用中位数代替
    data['NumberOfDependents'].fillna(data['NumberOfDependents'].mode()[0], inplace=True)
    # data = data.dropna()  # 或者直接删除比较少的缺失值
    return data


def explore_analysis(df):
    """
    对数据进行过探索性分析，
    常用有：
    直方图、散点图和箱线图等
    :param df:
    :return:
    """
    # 相关性分析
    cor = df.corr()
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns, annot=True, ax=ax)
    plt.show()
    # 面利用直方图和核密度估计画图
    fig = plt.figure()
    fig.set(alpha=0.2)
    plt.subplot2grid((2, 3), (0, 0))
    df['age'].plot(kind='hist', bins=30, figsize=(12, 6), grid=True)
    plt.title('Hist of Age')

    plt.subplot2grid((2, 3), (0, 1))
    df['age'].plot(kind='kde', figsize=(12, 6), grid=True)
    plt.title('KDE of Age')

    plt.subplot2grid((2, 3), (0, 2))
    df['MonthlyIncome'].plot(kind='kde', figsize=(12, 6), grid=True)
    plt.xlim(-20000, 80000)
    plt.title('KDE of MonthlyIncome')

    plt.subplot2grid((2, 3), (1, 0))
    df['NumberOfDependents'].plot(kind='kde', figsize=(12, 6), grid=True)
    plt.title('KDE of NumberOfDependents')

    plt.subplot2grid((2, 3), (1, 1))
    df['NumberOfOpenCreditLinesAndLoans'].plot(kind='kde', figsize=(12, 6), grid=True)
    plt.title('KDE of NumberOfOpenCreditLinesAndLoans')

    plt.subplot2grid((2, 3), (1, 2))
    df["NumberRealEstateLoansOrLines"].plot(kind="kde")
    plt.title("KDE of NumberRealEstateLoansOrLines")

    # 解决中文的显示问题
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.tight_layout()  # 调整子图之间的间距，紧凑显示图像
    plt.show()


def check_outlier(df):
    """
    异常值是指明显偏离大多数抽样数据的数值
    箱线图 可以确认是否存在异常值
    :param df:
    :return:
    """
    fig = plt.figure()
    fig.set(alpha=0.9)
    # plt.subplot2grid((5, 2), (0, 0))
    df['RevolvingUtilizationOfUnsecuredLines'].plot(kind='box', figsize=(8, 10), grid=True)
    plt.show()
    # plt.subplot2grid((5, 2), (0, 1))
    df['age'].plot(kind='box', figsize=(8, 10), grid=True)
    plt.show()
    # plt.subplot2grid((5, 2), (1, 0))
    df['NumberOfTime30-59DaysPastDueNotWorse'].plot(kind='box', figsize=(8, 10), grid=True)
    plt.show()
    # plt.subplot2grid((5, 2), (1, 1))
    df['DebtRatio'].plot(kind='box', figsize=(8, 10), grid=True)
    plt.show()
    # plt.subplot2grid((5, 2), (2, 0))
    df['MonthlyIncome'].plot(kind='box', figsize=(8, 10), grid=True)
    plt.show()
    # plt.subplot2grid((5, 2), (2, 1))
    df['NumberOfOpenCreditLinesAndLoans'].plot(kind='box', figsize=(8, 10), grid=True)
    plt.show()
    # plt.subplot2grid((5, 2), (3, 0))
    df['NumberOfTimes90DaysLate'].plot(kind='box', figsize=(8, 10), grid=True)
    plt.show()
    # plt.subplot2grid((5, 2), (3, 1))
    df['NumberRealEstateLoansOrLines'].plot(kind='box', figsize=(8, 10), grid=True)
    plt.show()
    # plt.subplot2grid((5, 2), (4, 0))
    df['NumberOfTime60-89DaysPastDueNotWorse'].plot(kind='box', figsize=(8, 10), grid=True)
    plt.show()
    # plt.subplot2grid((5, 2), (4, 1))
    df['NumberOfDependents'].plot(kind='box', figsize=(8, 10), grid=True)
    plt.show()

    """
    2019-09-12 16:34:52,105 : INFO : 查看NumberOfTime30-59DaysPastDueNotWorse共有哪些唯一值，判断哪个是异常值
    [ 2  0  1  3  4  5  7 10  6 98 12  8  9 96 13 11]
    2019-09-12 16:34:52,105 : INFO : 查看NumberOfTime60-89DaysPastDueNotWorse共有哪些唯一值，判断哪个是异常值
    [ 0  1  2  5  3 98  4  6  7  8 96 11  9]
    2019-09-12 16:34:52,105 : INFO : 查看NumberOfTimes90DaysLate共有哪些唯一值，判断哪个是异常值
     [ 0  1  3  2  5  4 98 10  9  6  7  8 15 96 11 13 14 17 12]
     存在两个近100 的异常值
    """


def replace_outlier(df, col):
    mode = df[col].mode()[0]
    df.loc[df[col] > 90, col] = mode
    logging.info(
        f'查看{col}共有哪些唯一值，判断哪个是异常值\n {df[col].unique()}')
    return df


def deal_abnormal(df):  # 异常值处理
    df = replace_outlier(df, 'NumberOfTime30-59DaysPastDueNotWorse')
    df = replace_outlier(df, 'NumberOfTime60-89DaysPastDueNotWorse')
    df = replace_outlier(df, 'NumberOfTimes90DaysLate')
    logging.info(
        f'处理后NumberOfTime30-59DaysPastDueNotWorse共有哪些唯一值，判断哪个是异常值\n {df["NumberOfTime30-59DaysPastDueNotWorse"].unique()}')
    logging.info(
        f'处理后NumberOfTime60-89DaysPastDueNotWorse共有哪些唯一值，判断哪个是异常值\n {df["NumberOfTime60-89DaysPastDueNotWorse"].unique()}')
    logging.info(f'处理后NumberOfTimes90DaysLate共有哪些唯一值，判断哪个是异常值\n {df["NumberOfTimes90DaysLate"].unique()}')


def split_data(df):
    """
    切分数据集，80%training，20% validation
    :param df:
    :return:
    """
    y = df['SeriousDlqin2yrs']
    X = df.iloc[:, 1:]
    # 测试和训练数据进行3：7的比例进行切分 random_state定一个值是的每次运行的时候不会被随机分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    train = pd.concat([y_train, X_train], axis=1)
    test = pd.concat([y_test, X_test], axis=1)
    train.to_csv('train_data.csv', index=False)
    test.to_csv('test_data.csv', index=False)


def feature_selection(Y, X, n=20):
    """
    特征选择：
    变量分箱：
        将连续变量离散化
        将多状态的离散变量合并成少状态
    变量分箱的重要性：
        1、稳定性：避免特征中无意义的波动对评分带来波动
        2、健壮性：避免极端值的影响
    变量分箱的优势：
        1、可以将缺失值作为一个独立的箱带入模型中
        2、将所有的变量变换到相似的尺度上
    变量分箱常用的方法：
     有监督的：
        1、Best-KS； 2、ChiMerge（卡方分箱法）
     无监督的：
        1、等距； 2、等频；(等频划分：将数据分成几等份，每等份数据里面的个数是一样) 3、聚类 (最优分段)
    :param df:
    :return:
    """
    # 首先选择对连续变量进行最优分段，在连续变量的分布不满足最优分段的要求时，再考虑对连续变量进行等距分段
    # 定义自动分箱函数
    from scipy.stats import stats
    r = 0
    good = Y.sum()
    bad = Y.count() - good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)  # 使用斯皮尔曼等级相关系数来评估两个变量之间的相关性
        n -= 1
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_values(by='min'))
    logging.info(f'd4 value {d4}')
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n + 1):
        qua = X.quantile(i / (n + 1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(2))
    return d4, iv, cut, woe


if __name__ == '__main__':
    # 定义相关的路径
    root = Path.cwd().parent
    training_path = root / 'input' / 'GiveMeSomeCredit' / 'cs-training.csv'
    testing_path = root / 'input' / 'GiveMeSomeCredit' / 'cs-testing.csv'
    training = load_data(training_path)
    training = set_missing(training)  # 用随机森林填补比较多的缺失值
    training = training.drop_duplicates()  # 删除重复项
    # check_outlier(training)
    deal_abnormal(training)
    explore_analysis(training)
    # # miss_data = output / 'MissingData.csv'
    # # data.to_csv(miss_data, index=False)
    # deal_abnormal(data)
