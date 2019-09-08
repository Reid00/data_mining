"""
借助于NBA球员的命中率和罚球命中率两个来给各位球员做一次“人以群分”的效果。
数据来源于虎扑体育
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def parse_content():
    """
    从网页读取数据
    :return： 球员信息datafrom 数据
    """
    table = []
    for i in np.arange(1, 7):
        table.append(pd.read_html(f'https://nba.hupu.com/stats/players/pts/{i}')[0])

    # 将所有数据纵向合并为datafram
    players = pd.concat(table)
    # 变量重命名
    columns = ['排名', '球员', '球队', '得分', '命中-出手', '命中率', '命中-三分', '三分命中率', '命中-罚球', '罚球命中率', '场次', '上场时间']
    players.columns = columns
    # 删除标签为0 的记录
    players.drop(0, inplace=True)
    logging.info(f'处理前：\n{players.head()}')
    # 数据类型转化
    players['得分'] = players['得分'].astype('float')
    players['命中率'] = players['命中率'].str[:-1].astype('float') / 100
    players['三分命中率'] = players['三分命中率'].str[:-1].astype('float') / 100
    players['罚球命中率'] = players['罚球命中率'].str[:-1].astype('float') / 100
    players['场次'] = players['场次'].astype('int')
    players['上场时间'] = players['上场时间'].astype('float')

    logging.info(f'选择信息前5条： \n {players.head()}')
    return players


def kmeans(players):
    """
    使用球员的命中率和罚球命中率来做聚类
    进行聚类分析
    :return:
    """
    # 中文和符号正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置绘图风格
    plt.style.use('ggplot')

    # 绘制得分与命中率的散点图
    players.plot(x='罚球命中率', y='命中率', kind='scatter')
    plt.show()
    # 选择最佳的K值
    X = players[['罚球命中率', '命中率']]
    K = range(1, int(np.sqrt(players.shape[0])))
    GSSE = []
    for k in K:
        SSE = []
        kmeans = KMeans(n_clusters=k, random_state=10)
        kmeans.fit(X)
        # 聚类的标签结果
        labels = kmeans.labels_
        # 聚类中心
        centers = kmeans.cluster_centers_
        for label in set(labels):
            SSE.append(np.sum(np.sum((players[['罚球命中率', '命中率']].loc[labels == label] - centers[label, :]) ** 2)))
        GSSE.append(np.sum(SSE))

    # 显示K的个数与GSSE 的关系
    plt.plot(K, GSSE, 'b*-')
    plt.xlabel('聚类个数')
    plt.ylabel('簇内离差平方和')
    plt.title('选择最优的聚类个数')
    plt.show()  # 当k为7时，看上去簇内离差平方和之和的变化已慢慢变小，那么，我们不妨就将球员聚为7类

    num_cluster = 6
    kmeans = KMeans(n_clusters=num_cluster, random_state=1)
    kmeans.fit(X)
    # 聚类结果标签
    players['cluster'] = kmeans.labels_
    # 聚类中心
    centers = kmeans.cluster_centers_

    # 绘制散点图
    plt.scatter(x=X.iloc[:, 0], y=X.iloc[:, 1], c=players['cluster'], s=50, cmap='rainbow')
    plt.scatter(centers[:, 0], centers[:, 1], c='k', marker='*', s=100)
    plt.xlabel('罚球命中率')
    plt.ylabel('命中率')
    plt.show()


if __name__ == '__main__':
    players = parse_content()
    kmeans(players)
