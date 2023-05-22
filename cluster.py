import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_columns', 200)  # 设置显示列数
pd.set_option('display.max_rows', 100)  # 设置显示行数
colors = ['r', 'g', 'b', 'c', 'y']
markers = ['s', 'x', 'o', 'v', '1']

df = pd.read_excel('./temp/LRFMC.xls')
#df = (df - df.mean(axis=0)) / (df.std(axis=0))  # 数据标准化

# SSE = []
# scores = []
#
# for k in range(3, 10):
#     estimator = KMeans(n_clusters=k)  # 构造聚类器
#     estimator.fit(df)
#     #SSE.append(estimator.inertia_)  # estimator.inertia_获取聚类准则的总和
#     score = metrics.silhouette_score(df, estimator.labels_, metric='euclidean')
#     print(score)
#     scores.append(score)
# X = range(3, 10)
# plt.xlabel('k')
# plt.ylabel('轮廓系数')
# plt.plot(X, scores, 'o-')
# plt.title("K-Means 轮廓系数")
# plt.show()

model = KMeans(n_clusters=5)
model.fit(df)
# print(model.inertia_)

# 统计聚类类别的数目和聚类中心
cluster_count = pd.Series(model.labels_).value_counts()
cluster_center = pd.DataFrame(model.cluster_centers_)
# print(cluster_count, cluster_center)

# 所有簇中心坐标值中最大值和最小值
max = cluster_center.values.max()
min = cluster_center.values.min()
cluster_info = pd.concat([cluster_center, cluster_count], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
cluster_info.columns = list(df.columns) + [u'类别数目']  # 重命名表头
# print(cluster_info)

# 绘图
fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111, polar=True)
center_num = cluster_info.values
feature = ["入会时间(L)", "最近乘机间隔(R)", "飞行次数(F)",  "飞行里程(M)",  "平均折扣率(C)"]
N = len(feature)
# print(center_num)

feature = np.concatenate((feature, [feature[0]]))
for i, v in enumerate(center_num):
    # 设置雷达图的角度，用于平分切开一个圆面
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # 为了使雷达图一圈封闭起来，需要下面的步骤
    center = np.concatenate((v[:-1], [v[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    # 绘制折线图
    ax.plot(angles, center, 'o-', linewidth=2, label="第%d簇客户群:%d人" % (i + 1, v[-1]))
    # 填充颜色
    ax.fill(angles, center, alpha=0.25)
    # 添加每个特征的标签
    print(feature)
    ax.set_thetagrids(angles * 180 / np.pi, feature, fontsize=15)
    # 设置雷达图的范围
    ax.set_ylim(min - 0.1, max + 0.1)
    # 添加标题
    plt.title('客户群特征分析图', fontsize=20)
    # 添加网格线
    ax.grid(True)
    # 设置图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), ncol=1, fancybox=True, shadow=True)
# 显示图形
plt.savefig("./temp/kmeans.png")
plt.show()






