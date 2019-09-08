from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def exercise():
    """方法fit_predict的作用是计算聚类中心,并为输入的数据加上分类标签
    fit方法的使用，它仅仅产生聚类中心（其实也就是建模），然后我们引入两个新的点，并利用已经建立的模型预测它们的分类情况
    """
    # 获取模拟数据
    x = np.random.rand(100, 2)

    # n_clusters 表示聚类数目，random_state 产生随机数的方法
    # y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(x)
    # 聚类聚为3类
    estimator = KMeans(n_clusters=3)
    # fit_predict表示拟合+预测
    res = estimator.fit_predict(x)
    # 预测类别标签结果
    label_pred = estimator.labels_
    # 各个类别的聚类中心值
    centroids = estimator.cluster_centers_
    # 聚类中心均值向量的总和
    inertia = estimator.inertia_

    #
    logging.info(label_pred)
    logging.info(centroids)
    logging.info(inertia)

    # 通过图形化展示聚类效果
    for i in range(len(x)):
        if int(label_pred[i]) == 0:
            plt.scatter(x[i][0], x[i][1], color='red', marker='x')
        if int(label_pred[i]) == 1:
            plt.scatter(x[i][0], x[i][1], color='black', marker='s')
        if int(label_pred[i]) == 2:
            plt.scatter(x[i][0], x[i][1], color='blue', marker='o')
    plt.show()

    # fit method
    # 训练
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x)

    new_data = np.array([[3, 3],
                         [15, 15]])
    # 预测
    color = ('red', 'yellow')
    colors_2 = np.array(color)[kmeans.predict([[3, 3], [15, 15]])]
    plt.scatter(new_data[:, 0], new_data[:, 1], c=colors_2, marker='x')
    plt.show()


if __name__ == '__main__':
    exercise()
