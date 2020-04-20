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

### 第二种详细训练方式

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import lightgbm as lgbm

def load_data(path):
    data = pd.read_csv(path,sep=',',encoding='utf-8',parse_dates=['Dates'])
    return data

# 根据时间进行季节划分
def season(month):
    if (month<3 or month>=12):
        return 1  # 冬季
    if (month >=3 and month < 6):
        return 2  # 春季
    if (month >=6 and month <9):
        return 3  # 夏季
    if (month >=9 and month < 12):
        return 4  # 秋季

# 根据时间进行时段的划分
def hour_box(hour):
    if (hour>=1 and hour <8 ):
        return 1
    if (hour >=8 and hour<12):
        return 2
    if (hour>=12 and hour<13):
        return 3
    if (hour>=13 and hour<18):
        return 4
    if (hour >= 18 and hour <20):
        return 5
    if (hour >= 20 or hour <1):
        return 6

# 地址中是否包含 /
def address_type(addr):
    if '/'in addr:
        return 0
    else:
        return 1

# 地址是否包含门牌号
def address_num(addr):
    pattern = re.compile(r'\d+')
    res=re.findall(pattern,addr)
    if len(res)==0:
        return 0
    else:
        return res[0]

def feature_engineer(data):
    le = LabelEncoder()
    # onehot = OneHotEncoder(sparse=False)
    #去重
    print('重复数据的数量为:',data.duplicated().sum())
    data.drop_duplicates(inplace=True)
    print('去重后的数量为:',data.shape[0])

    # 将目标变量设置为二分类
    data.loc[data['Category']=='LARCENY/THEFT','Cates'] = 1
    data.loc[data['Category']=='OTHER OFFENSES','Cates'] = 0

    # 根据时间进行季节划分
    data['Year']=data['Dates'].dt.year
    data['Month']=data['Dates'].dt.month
    data['Day']=data['Dates'].dt.day
    data['Hour']=data['Dates'].dt.hour
    data['Weekofyear']=data['Dates'].dt.weekofyear

    # 获取季节信息
    data['Season'] = data['Month'].apply(season)
    # onehot.fit(data['Season'].values.reshape(-1,1))
    # data['Season_encoder'] = onehot.transform(data['Season'])
    # 增加时段特征
    data['Hour_box'] =data['Hour'].apply(hour_box)
    # data['Hour_box_encoder']= onehot.fit_transform(data['Hour_box'].values.reshape(-1,1))
    # dayofweek 进行梳理
    data['DayOfWeek_encoder'] = le.fit_transform(data['DayOfWeek'])
    # data['DayOfWeek_encoder'] = onehot.fit_transform(data['DayOfWeek_encoder'].values.reshape(-1,1))

    #是否是周末
    weekend = weekend= {'Monday':0., 'Tuesday':0., 'Wednesday':0., 'Thursday': 0., 'Friday':0., 'Saturday':1., 'Sunday':1}
    data['Weekend'] = data['DayOfWeek'].replace(weekend)

    # PdDistrict,Descript 数据编码
    data['PdDistrict_encoder'] = le.fit_transform(data['PdDistrict'])
    data['Descript_encoder'] = le.fit_transform(data['Descript'])

    # 地址相关信息处理
    data['Addr_type'] = data['Address'].apply(address_type)
    # data['Addr_type'] = onehot.fit_transform(data['Addr_type'].values.reshape(-1,1))
    data['Addr_num'] =data['Address'].apply(address_num)

    # 二分类的数据
    data = data.loc[(data['Cates']==1) | (data['Cates']==0)]

    return data

def model_training(data):
    # 将数据划分为训练集和测试集
    X=data.drop(columns=['Dates','Cates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','Hour','Season','Hour_box'])
    y = data['Cates']
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
    rfc = RandomForestClassifier(n_estimators=200,oob_score=True,min_samples_leaf=200,min_samples_split=200)
    print(r'training model, please waiting...')
    rfc.fit(X_train.values,y_train.values)

    y_predprob = rfc.predict_proba(X_test.values)
    y_pred= rfc.predict(X_test.values)

    # log_loss = sklearn.metrics.log_loss(y_test,y_predprob)
    cm = confusion_matrix(y_test,y_pred)
    cr = classification_report(y_test,y_pred)
    auc = roc_auc_score(y_test,y_pred)

    importance= rfc.feature_importances_
    features = sorted(zip(map(lambda x: round(x,4),importance),X_train.columns.tolist()))
    print('随机森林模型的特征排名:',features)
    # print('随机森林模型训练的log 损失值为:',log_loss)
    print('随机森林模型的cm:\n',cm)
    print('随机森林模型的cr:\n',cr)
    print('随机森林模型的auc:\n',auc)
    print('随机森林模型的oob_score:',rfc.oob_score_)

def main():
    root = Path.cwd()
    path= root / r'data_mining-master\input\sf-crime\train.csv'
    data=load_data(path)
    binary_data=feature_engineer(data)
    model_training(binary_data)

if __name__ == "__main__":
    main()
