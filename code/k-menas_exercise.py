from sklearn import cluster
from matplotlib import pyplot as plt
from matplotlib import image
from pathlib import Path

root = Path.cwd()
input = root / '..' / 'input'
output = root / '..' / 'output'

# 获取image 路径
img_paths = input.glob('*.png')

# 读取图片
for img_path in img_paths:
    img = image.imread(r'D:\v-baoz\python\data_mining\input\img.png')
    # img = image.imread(img_path)
    # 将图片转换为像素向量
    orig_shape = img.shape
    # img = img.reshape((-1, 3))
    plt.close()
    plt.axis('off')
    plt.imshow(img.reshape(orig_shape))
    orig = output / 'orig.png'
    plt.savefig(orig)

    for k in range(2, 10):
        # 创建运行k-means 算法对象
        kmeans_fitter = cluster.KMeans(n_clusters=k)
        # 运行k-means 算法
        kmeans_fitter.fit(img)
        # 得到每个像素所在的组
        kmeans_grp = kmeans_fitter.labels_
        # 得到每个聚类的中心
        kmeans_centroids = kmeans_fitter.cluster_centers_
        # 将每个像素替换为聚类中心
        result = kmeans_centroids[kmeans_grp]
        # 显示图片
        plt.close()
        plt.axis('off')
        plt.imshow(result.reshape(orig_shape))
        transformed = output / f'transformed_{(k,)}.png'
        plt.savefig(transformed)
