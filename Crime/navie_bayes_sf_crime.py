"""
朴素贝叶斯应用场景：
1.文本分类/垃圾文本过滤/情感判别，同时在文本数据中，分布独立这个假设基本是成
2.多分类实时预测：这个是不是不能叫做场景？对于文本相关的多分类实时预测，它因为上面提到的优点，被广泛应用，简单又高效。
3. 推荐系统：是的，你没听错，是用在推荐系统里！！朴素贝叶斯和协同过滤(Collaborative Filtering)是一对好搭档，协同过滤是强相关性，但是泛化能力略弱，朴素贝叶斯和协同过滤一起，能增强推荐的覆盖度和效果。

"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
旧金山犯罪分类预测
"""


def load_data(path):
    df = pd.read_csv(path, encoding='utf8', parse_dates=['Dates'])
    logging.info(f'{path.name} 前五行的数据为\n {df.head()}')
    return df


"""
Date: 日期
Category: 犯罪类型，比如 Larceny/盗窃罪 等.
Descript: 对于犯罪更详细的描述
DayOfWeek: 星期几
PdDistrict: 所属警区
Resolution: 处理结果，比如说『逮捕』『逃了』
Address: 发生街区位置
X and Y: GPS坐标
"""


def feature_engineer(df):
    """
    特征工程预处理
    用LabelEncoder对犯罪类型做编号
    对街区，星期几，时间点用get_dummies()因子化
    做一些组合特征，比如把上述三个feature拼在一起，再因子化一下；
    :param df:
    :return:
    """
    # 用LabelEncoder对不同的犯罪类型编号
    le = preprocessing.LabelEncoder()
    crime = le.fit_transform(df['Category']) if 'Category' in df.columns else None

    # 因子化星期几，街区，小时等特征
    days = pd.get_dummies(df['DayOfWeek'])
    district = pd.get_dummies(df['PdDistrict'])
    hour = df['Dates'].dt.hour
    hour = pd.get_dummies(hour)

    # 组合特征
    data = pd.concat([hour, days, district], axis=1)
    data['crime'] = crime
    logging.info(f'前五行数据为\n {data.head()}')
    return data


def navie_bayes(df):
    """
    朴素贝叶斯进行预测
    :param df:
    :return:
    """
    # # 只取星期几和街区作为分类器输入特征
    features = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'BAYVIEW', 'CENTRAL',
                'INGLESIDE', 'MISSION',
                'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
    # 添加犯罪的小时时间点作为特征
    hour_fea = [x for x in np.arange(0, 24)]
    features = features + hour_fea

    # 分割训练集(70%)和测试集(30%)
    training, validtion = train_test_split(df, test_size=0.3)
    # 朴素贝叶斯建模，计算log_loss
    model = BernoulliNB()
    start = time.time()
    model.fit(training[features], training['crime'])
    cost_time = time.time() - start
    pridected = np.array(model.predict_proba(validtion[features]))
    loss = log_loss(validtion['crime'], pridected)
    logging.info(f'朴素贝叶斯建模耗时{cost_time}秒')
    logging.info(f'朴素贝叶斯log 损失为{loss}')


def logistic(df):
    """
    逻辑回归建模
    :param df:
    :return:
    """
    # 只取星期几和街区作为分类器输入特征
    features = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'BAYVIEW', 'CENTRAL',
                'INGLESIDE', 'MISSION',
                'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
    # 分割训练集(70%)和测试集(30%)
    training, validtion = train_test_split(df, test_size=0.3)
    # 逻辑回归建模，计算log_loss
    model = LogisticRegression(C=0.01, solver='lbfgs', multi_class='auto', max_iter=10000)
    start = time.time()
    model.fit(training[features], training['crime'])
    cost_time = time.time() - start
    pridected = np.array(model.predict_proba(validtion[features]))
    loss = log_loss(validtion['crime'], pridected)
    logging.info(f'逻辑回归建模耗时{cost_time}秒')
    logging.info(f'逻辑回归log 损失为{loss}')


"""
2019-09-05 16:34:49,843 : INFO : 朴素贝叶斯建模耗时0.9860095977783203秒
2019-09-05 16:34:49,844 : INFO : 朴素贝叶斯log 损失为2.581347846129299
2019-09-05 16:45:00,622 : INFO : 逻辑回归建模耗时609.3348324298859秒
2019-09-05 16:45:00,622 : INFO : 逻辑回归log 损失为2.6156089163206016
逻辑回归建模耗时太长，舍弃
"""

if __name__ == '__main__':
    root = Path.cwd().parent
    train = root / 'input' / 'sf-crime' / 'train.csv'
    test = root / 'input' / 'sf-crime' / 'test.csv'
    train = load_data(train)
    test = load_data(test)
    train_data = feature_engineer(train)
    test_data = feature_engineer(test)
    navie_bayes(train_data)
    # logistic(train_data)
