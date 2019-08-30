from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pd.set_option('display.max_columns', None)
"""
基于Python的信用评分模型开发.
信用评分模型可用“四张卡”来表示，分别是 A卡（Application score card，申请评分卡）
B卡（Behavior score card，行为评分卡）
C卡（Collection score card，催收评分卡）
F卡（Anti-Fraud Card，反欺诈评分卡），分别应用于贷前、贷中、贷后。

"""
# 定义相关的路径
root = Path.cwd()
output = root.parent / 'output' / 'credit'


def parse_data():  # 数据预处理
    cs_file = root.parent / 'input' / 'GiveMeSomeCredit' / 'cs-training.csv'
    data = pd.read_csv(cs_file, encoding='utf8')
    logging.info(f'前五条数据{data.head()}')
    desc = output / 'data_desc.csv'
    data.describe().to_csv(desc)
    logging.info(f'NA值的个数{data.isnull().sum()}')
    return data


def set_missing(data):  # 缺失值处理
    # １．直接删除含有缺失值的样本。
    # ２．根据样本之间的相似性填补缺失值。
    # ３．根据变量之间的相关关系填补缺失值。
    # MonthlyIncome 缺失率比较大变量之间的相关关系填补缺失值
    # 用随机森林对缺失值预测填充函数
    # 把已有的数值型特征取出来
    process_df = data.iloc[:, [6, 0, 1, 2, 3, 4, 5, 7, 8, 9]]
    # 分成已知该特征和未知该特征两部分
    known = process_df[process_df.MonthlyIncome.notnull()].values
    unknown = process_df[process_df.MonthlyIncome.isnull()].values
    # x 为特征属性值
    x = known[:, 1:]
    # y 为结果标签值
    y = known[:, 0]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(x, y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    logging.info(f'预测结果{predicted}')
    # 用得到的预测结果填补原缺失数据
    data.loc[data['MonthlyIncome'].isnull(), 'MonthlyIncome'] = predicted
    # NumberOfDependents  变量缺失值比较少，可以用均值代替
    data['NumberOfDependents'].fillna(data['NumberOfDependents'].mode()[0], inplace=True)
    # data = data.dropna()  # 或者直接删除比较少的缺失值
    return data


def deal_abnormal(data):  # 异常值处理
    # 异常值是指明显偏离大多数抽样数据的数值
    data = data[data['age'] > 0]  # 年龄等于0的异常值进行剔除
    # 箱线图 可以确认是否存在异常值
    data.iloc[:, [2, 5, 6]].boxplot()
    plt.show()


if __name__ == '__main__':
    data = parse_data()
    # NumberOfDependents  变量缺失值比较少，直接删除，对总体模型不会造成太大影响
    data = set_missing(data)  # 用随机森林填补比较多的缺失值
    data = data.drop_duplicates()  # 删除重复项
    # miss_data = output / 'MissingData.csv'
    # data.to_csv(miss_data, index=False)
    deal_abnormal(data)
