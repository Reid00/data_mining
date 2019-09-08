import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
"""
此文件作为练习，机器学习问题解决思路
拿到数据后怎么了解数据(可视化)
选择最贴切的机器学习算法
定位模型状态(过/欠拟合)以及解决方法
大量极的数据的特征分析与可视化
各种损失函数(loss function)的优缺点及如何选择
"""


def gen_data():
    """
    使用make_classification构造1000个样本，每个样本有20个feature
    :return:
    """
    X, y = make_classification(1000, n_features=20, n_informative=2, n_redundant=2, n_classes=2, random_state=0)
    # logging.debug(X, y)
    # print(len(y))
    df = pd.DataFrame(np.hstack((X, y[:, None])), columns=range(21))
    logging.info(f'打印前五行数据:\n {df.head()}')
    return df


def visual_analyze(df):
    """
    数据可视化分析
    :param df:
    :return:
    """
    # 使用pairplot去看不同特征维度pair下数据的空间分布状况
    _ = sns.pairplot(df[:50], vars=[8, 11, 12, 14, 19], height=1.5)
    plt.show()

    # 用Seanborn中的corrplot来计算计算各维度特征之间(以及最后的类别)的相关性
    # corrplot 被删除，可以使用heatmap ，方法如下：
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


if __name__ == '__main__':
    df = gen_data()
    visual_analyze(df)
