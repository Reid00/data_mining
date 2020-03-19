
"""
kaggle-美国人口普查年收入比赛
"""
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path


def read_data(path):
    columns=['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus','Occupation','Relationship','Race','Sex','CapitalGain','CapitalLoss','HoursPerWeek','Country','Income']
    data=pd.read_csv(path,sep=',',names=columns)
    # train_data.columns=columns
    # print(train_data.columns)
    print(data.head(5))
    return data

def show_data_info(DataFrame):
    data= DataFrame
    print(f'data shape is: \n {data.shape}')
    print(f'data infomation is: \n {data.info()}')
    print(f'data describe is: \n {data.describe()}')
    print(f'data isnull is: \n {data.isnull().sum()}')
    # print(f'data notnull is: \n {data.notnull().sum()}')
    print(f'data age values: \n {data["Age"].value_counts()}')
    print(f'data Workclass values: \n {data["Workclass"].value_counts()}')
    print(f'data Income values: \n {data["Income"].value_counts()}')

def feature_engineer(DataFrame):
    data=DataFrame
    # data=data.drop(columns='fnlgwt')
    print(data.replace('?',np.nan).shape)
    print(data.replace('?',np.nan).dropna().shape)
    # 删除含有?（缺失行）
    data= data.replace('?',np.nan).dropna()

    # 把测试语料集中的 Income 归一化
    data['Income']=data['Income'].apply(lambda val: val.strip().replace('.',''))
    # data['Income']=data['Income'].str.strip().replace({'<=50K.':'<=50K','>50K.':'>50K'})
    print(data['Income'].unique())
    print(f'data Income values: \n {data["Income"].value_counts()}')
    
    # 因为有 受教育的年数，所以这里不需要教育这一列
    data=data.drop(columns='Education')
    # Age 和 EdNum 列是数值型的，我们可以将连续数值型转化为更高效的方式，例如将年龄换为 10 年的整数倍，教育年限换为 5 年的整数倍，实现的代码如下：
    labels=[f'{i}-{i+9}' for i in range(0,100,10)]
    data['Age_group']=pd.cut(data['Age'], range(0,101,10), right=False, labels=labels)
    edu_labels=[f'{i}-{i+4}' for i in range(0,20,5)]
    data['Edu_group']=pd.cut(data['EdNum'],range(0,21,5),right=False, labels=edu_labels)
    data.drop(columns=['Age','EdNum'],inplace=True)
    
    print('{}'.format(data.info()))
    le=LabelEncoder()
    # 将表中的object数据转化为int类型
    print('start label encoding: ')
    for feature in data.columns:
        if data[feature].dtype != 'int64':
            le.fit(data[feature])
            print(f'feature name is: {feature}')
            data[feature]= le.fit_transform(data[feature])
            # print(data[feature])
            # data[feature] = pd.Categorical(data[feature]).codes

    X_train= data.drop(columns='Income')
    print(X_train.info())
    y= data['Income']
    print('feature engineer done.')
    return X_train,y

def model_pred(X_train,y_train,X_test,y_test):
    print('initial model training')
    classifier= RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=0)
    print('start training')
    classifier.fit(X_train.values,y_train.values)

    # 预测测试集结果
    print('start predict')
    y_pred=classifier.predict(X_test)

    print('start confusion matrix')
    cm = confusion_matrix(y_test,y_pred)
    cr= classification_report(y_test,y_pred)
    # 打印混淆矩阵 cm 以及分析结果（准确率，召回率，F1值）cr：
    print(f'cm: {cm}','\n',f'cr: {cr}')

    # 计算出ROC AUC
    roc_value = roc_auc_score(y_test, y_pred)
    print(f'roc_value is: \n {roc_value}')

if __name__ == "__main__":
    train_path=Path(r'D:\python\GitRepository\data_mining\input\RandomForest\adult.data')
    test_path=Path(r'D:\python\GitRepository\data_mining\input\RandomForest\adult.test')
    train_data=read_data(train_path)
    test_data=read_data(test_path)
    # show_data_info(train_data)
    X_train,y_train=feature_engineer(train_data)
    X_test,y_test=feature_engineer(test_data)
    model_pred(X_train,y_train,X_test,y_test)