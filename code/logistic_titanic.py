import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as prepro
from sklearn import linear_model
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ROOT = Path.cwd().parent
train = ROOT / 'input' / 'titanic' / 'train.csv'
test = ROOT / 'input' / 'titanic' / 'test.csv'

pd.options.display.max_columns = 10


def analyse_data(path):
    """
    用于探索数据
    :return: 返回训练集的数据
    """
    data_train = pd.read_csv(path, sep=',')
    # logging.info(f'train data 的前五行:\n {data_train.head()}')
    logging.info(data_train.describe())
    logging.info(f'\n {data_train.isnull().sum()}')
    logging.info(data_train.info())

    return data_train


def property_info(data_train):
    """
    乘客各属性分布
    :return:
    """
    # 中文和符号正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色的alpha 参数

    plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里面分列出几个小图，2行3列，从0,0 开始绘图
    data_train['Survived'].value_counts().plot(kind='bar')  # 存活人员绘制柱状图
    plt.title(u'获救情况(1 为获救)')  # 标题
    plt.ylabel(u'人数')

    plt.subplot2grid((2, 3), (0, 1))
    data_train['Pclass'].value_counts().plot(kind='bar')
    plt.title(u'乘客等级分布')
    plt.ylabel(u'人数')

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data_train['Survived'], data_train['Age'])  # 绘制年龄和获救之间的散点图
    plt.title(u'按照年龄观察获救分布情况（1 为获救）')
    plt.ylabel(u'年龄')
    plt.grid(b=True, which='major', axis='y')

    plt.subplot2grid((2, 3), (1, 0), colspan=2)  # colspan，横跨列（意思是左右合并）
    data_train[data_train['Pclass'] == 1]['Age'].plot(kind='kde')  # 密度图,与直方图相关的一种类型图，是通过计算“可能会产生观测数据的连续概率分布的估计”而产生的
    data_train[data_train['Pclass'] == 2]['Age'].plot(kind='kde')
    data_train[data_train['Pclass'] == 3]['Age'].plot(kind='kde')
    plt.title(u'各等级的乘客年龄分布')
    plt.xlabel(u'年龄')
    plt.ylabel(u'密度')
    plt.legend([u'头等舱', u'2等舱', u'3等舱'], loc='best')  # legend 显示图例提示信息

    plt.subplot2grid((2, 3), (1, 2))
    data_train['Embarked'].value_counts().plot(kind='bar')
    plt.title(u'各登船口岸上船人数')
    plt.ylabel(u'人数')

    plt.show()

    """
    看看各乘客等级的获救情况
    """
    fig2 = plt.figure()
    fig2.set(alpha=0.2)

    survived_0 = data_train['Pclass'][data_train['Survived'] == 0].value_counts()
    survived_1 = data_train['Pclass'][data_train['Survived'] == 1].value_counts()
    df = pd.DataFrame({
        u'获救': survived_1,
        u'未获救': survived_0
    })
    df.plot(kind='bar', stacked=True)
    plt.title(u'各乘客等级的获救情况')
    plt.xlabel(u'乘客等级')
    plt.ylabel(u'人数')
    plt.show()

    """
    查看性别对获救情况的影响    
    """
    fig3 = plt.figure()
    fig3.set(alpha=0)  # 设定图表颜色alpha参数
    survived_m = data_train['Survived'][data_train['Sex'] == 'male'].value_counts()
    survived_f = data_train['Survived'][data_train['Sex'] == 'female'].value_counts()
    gender_df = pd.DataFrame({
        u'男性': survived_m,
        u'女性': survived_f
    })
    gender_df.plot(kind='bar', stacked=True)
    plt.title(u'按性别查看获救情况')
    plt.xlabel(u'性别')
    plt.ylabel(u'人数')
    plt.show()

    """
     然后我们再来看看各种舱级别情况下各性别的获救情况
    """
    fig4 = plt.figure()
    fig3.set(alpha=0.5)
    plt.title(u'船舱等级和性别的获救情况')
    ax1 = fig4.add_subplot(141)  # 参数意思是将画布分割成1行4列，图像在从左到右从上到下的第1块
    # data_train['Survived'][data_train['Sex'] == 'female'][data_train['Pclass'] != 3].value_counts().plot(
    #         kind='bar',
    #         label='female highclass',
    #         color='r')
    # 或者用这种方法过滤
    data_train.loc[(data_train['Sex'] == 'female') & (data_train['Pclass'] != 3), 'Survived'].value_counts().plot(
        kind='bar',
        label='female highclass',
        color='r')
    # # 或者使用透视图
    # pt = data_train.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc=['count'])
    # print(pt)
    ax1.set_xticklabels([u'获救', u'未获救'], rotation=0)
    plt.legend([u'女性/高级舱'], loc='best')

    ax2 = fig4.add_subplot(142, sharey=ax1)
    data_train.loc[(data_train['Sex'] == 'female') & (data_train['Pclass'] == 3), 'Survived'].value_counts().plot(
        kind='bar', label='female low class', color='pink')
    ax2.set_xticklabels([u'未获救', u'获救'], rotation=0)
    plt.legend([u'女性/低级舱'], loc='best')

    ax3 = fig4.add_subplot(143, sharey=ax1)
    data_train.loc[(data_train['Sex'] == 'male') & (data_train['Pclass'] != 3), 'Survived'].value_counts().plot(
        kind='bar', label='male,high class', color='lightblue')
    ax3.set_xticklabels([u'未获救', u'获救'], rotation=0)
    plt.legend([u'男性/高级舱'], loc='best')

    ax4 = fig4.add_subplot(144, sharey=ax1)
    data_train.loc[(data_train['Sex'] == 'male') & (data_train['Pclass'] == 3), 'Survived'].value_counts().plot(
        kind='bar', label='male,high class', color='steelblue')
    ax4.set_xticklabels([u'未获救', u'获救'], rotation=0)
    plt.legend([u'男性/低级舱'], loc='best')
    plt.show()

    """
    查看各个登船港口的获救情况
    """
    fig5 = plt.figure()
    fig5.set(alpha=0.2)

    # embarked_pt = data_train.pivot_table(index=['Embarked'], values='Survived', aggfunc='count')
    Survived_0 = data_train.loc[data_train['Survived'] == 0, 'Embarked'].value_counts()
    Survived_1 = data_train.loc[data_train['Survived'] == 1, 'Embarked'].value_counts()
    embarked_pt = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    embarked_pt.plot(kind='bar', stacked=True)
    plt.title(u'各登陆港口乘客的获救情况')
    plt.xlabel(u'登陆港口')
    plt.ylabel(u'人数')
    # plt.show()

    """
    观察堂兄弟姐妹，孩子，父母几人对获救是否有影响
    """
    grouped = data_train.groupby(['SibSp', 'Survived'])['PassengerId'].count()
    # pivot_table 方法
    pt_sibsp = data_train.pivot_table(index=['SibSp', 'Survived'], values='PassengerId', aggfunc='count')
    pt_parch = data_train.pivot_table(index=['Parch', 'Survived'], values='PassengerId', aggfunc='count')
    print(pt_parch)
    print(pt_sibsp)

    """
    ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把
    cabin只有204个乘客有值，我们先看看它的一个分布
    """
    data_train['Cabin'].value_counts()

    """
    在有无Cabin信息这个粗粒度上看看Survived的情况
    
    有Cabin记录的似乎获救概率稍高一些，先这么着放一放吧。
    """
    survived_cabin = data_train.loc[data_train['Cabin'].notnull(), 'Survived'].value_counts()
    survived_nocabin = data_train.loc[data_train['Cabin'].isnull(), 'Survived'].value_counts()
    df_cabin = pd.DataFrame({u'有': survived_cabin, u'无': survived_nocabin}).transpose()  # transpose 行列转化
    df_cabin.plot(kind='bar', stacked=True)
    plt.title(u'按Cabin有无查看获救情况')
    plt.xlabel(u'有无Cabin')
    plt.ylabel(u'人数')
    plt.show()


def set_missing(df):
    """
    用随机森林填缺失的年龄属性
    :param data_train:
    :return:
    """
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_data = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_data[age_data['Age'].notnull()].values
    unknown_age = age_data[age_data['Age'].isnull()].values

    # y 即是目标年龄
    y = known_age[:, 0]
    # x 即是特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    pred_age = rfr.predict(unknown_age[:, 1:])

    # 用得到的预测结果填补原缺失数据
    df.loc[age_data['Age'].isnull(), 'Age'] = pred_age

    return df, rfr


def set_Cabin_type(df):
    df['Cabin'] = np.where(df['Cabin'].isnull(), 'No', 'Yes')
    return df


def feature_factors(df):
    """
    因为逻辑回归建模时，需要输入的特征都是数值型特征，我们通常会先对类目型的特征因子化。
    :param df:
    :return:
    """
    dummies_cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummies_embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummies_sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummies_pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df = pd.concat([df, dummies_cabin, dummies_embarked, dummies_pclass, dummies_sex, dummies_pclass], axis=1)
    df.drop(columns=['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], inplace=True)
    print(df.head())
    return df


def scaling(df):
    """
    仔细看看Age和Fare两个属性，乘客的数值幅度变化
    各属性值之间scale差距太大，将对收敛速度造成几万点伤害值
    :param df:
    :return:
    """
    scaler = prepro.StandardScaler()
    age_scale_param = scaler.fit(df['Age'].values.reshape(1, -1))
    df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    return df


def logistic(df):
    """
    把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
    :param df:
    :return:
    """
    # 用正则取出我们要的属性值
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]
    # fit到RandomForestRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)
    print(clf)
    return clf


def test_data(rfr, clf):
    """
    处理测试集数据
    :param df:
    :return:
    """
    # 测试模型预估
    data_test = analyse_data(test)
    df_age = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = df_age.loc[df_age['Age'].isnull()].values
    X = null_age[:, 1:]
    pred_age = rfr.predict(X)
    data_test.loc[data_test['Age'].isnull(), 'Age'] = pred_age

    data_test = set_Cabin_type(data_test)

    data_test = feature_factors(data_test)

    data = data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(data)
    res = pd.DataFrame({
        'PassengerId': data_test['PassengerId'].values,
        'Survived': predictions.astype(np.float32)
    })
    output = ROOT / 'output' / 'titanic_res.csv'
    res.to_csv(output, index=False)


def optimizate_model(df):
    """
    逻辑回归系统优化
    :param df:
    :return:
    """
    # 交叉验证
    # train_test_split：把train.csv分成两部分，一部分用于训练我们需要的模型，另外一部分数据上看我们预测算法的效果。
    # 分割数据，按照 训练数据:cv数据 = 7:3的比例
    split_train, split_test = train_test_split(df, test_size=0.3, random_state=0)
    train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

    # 生成模型
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(train_df.values[:, 1:], train_df.values[:, 0])

    # 对train_test_split test数据进行预测
    test_df = split_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test_df.values[:, 1:])
    logging.info(f'pridictions{predictions}')
    origin_train_data = df
    bad_cases = origin_train_data.loc[
        origin_train_data['PassengerId'].isin(
            split_test.loc[predictions != test_df.values[:, 0], 'PassengerId'].values)]
    logging.info(bad_cases)
    return clf


if __name__ == '__main__':
    data_train = analyse_data(train)
    # property_info(data_train)
    data_train, rfr = set_missing(data_train)
    data_train = set_Cabin_type(data_train)
    data_train = feature_factors(data_train)
    # data_train = scaling(data_train)
    clf = logistic(data_train)
    test_data(rfr, clf)
    optimizate_model(data_train)
