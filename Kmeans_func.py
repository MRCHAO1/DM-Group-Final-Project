#D:\Users\Lenovo\anaconda3\envs\chatbot\Scripts\pyuic5.exe K-means.ui -o Kmeans_ui.py
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
from Kmeans_ui import Ui_kmeans
from qtpandas.models.DataFrameModel import DataFrameModel
import pandas as pd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class Kmeans(QWidget, Ui_kmeans):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.LB_radar.setScaledContents(True)
        self.PB_confirm.clicked.connect(self.process)
        self.PB_openfile.clicked.connect(self.readData)
        self.model = DataFrameModel()
        self.pandasTableWidget.setViewModel(self.model)

    def PB_set(self, bool):
        self.PB_openfile.setEnabled(bool)
        self.PB_predict.setEnabled(bool)
        self.PB_confirm.setEnabled(bool)
        self.PB_back.setEnabled(bool)
        self.PB_Kmeans.setEnabled(bool)


    def readData(self):
        # self.df = pd.read_excel('./temp/kmeans.xls')
        # self.df_original = self.df.copy()  # 备份原始数据
        # self.model.setDataFrame(self.df_original.head(40))
        self.LB_stat.setText("读取中")
        self.PB_set(False)
        self.readDataThread = readThread()
        self.readDataThread.finishSignal.connect(self.readData_change)
        self.readDataThread.start()

    def readData_change(self, df):
        self.LB_stat.setText("读取完毕")
        self.PB_set(True)
        self.df = df.copy()  # 备份原始数据
        self.model.setDataFrame(self.df.head(40))

    def process(self):
        self.LB_stat.setText("处理中")
        self.PB_set(False)
        self.processThread = processThread(int(self.CB_k.currentText()))
        self.processThread.finishSignal.connect(self.process_change)
        self.processThread.start()

    def process_change(self, SSE_score):
        self.LB_stat.setText("处理完毕")
        self.PB_set(True)
        self.LE_SSE.setText(str(SSE_score[0])[:9])
        self.LE_Cont.setText(str(SSE_score[1])[:10])
        pix = QtGui.QPixmap('./temp/kmeans.png')
        self.LB_radar.setPixmap(pix)




class processThread(QThread):
    finishSignal = pyqtSignal(tuple)
    def __init__(self, k, parent=None):
        super(processThread, self).__init__(parent)
        self.k = k
    def run(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        pd.set_option('display.max_columns', 200)  # 设置显示列数
        pd.set_option('display.max_rows', 100)  # 设置显示行数
        colors = ['r', 'g', 'b', 'c', 'y']
        markers = ['s', 'x', 'o', 'v', '1']
        df = pd.read_excel('./temp/data_kmeans.xls')
        df = (df - df.mean(axis=0)) / (df.std(axis=0))
        model = KMeans(n_clusters=self.k)
        model.fit(df)
        SSE = model.inertia_  # estimator.inertia_获取聚类准则的总和
        score = metrics.silhouette_score(df, model.labels_, metric='euclidean')
        cluster_count = pd.Series(model.labels_).value_counts()
        cluster_center = pd.DataFrame(model.cluster_centers_)
        max = cluster_center.values.max()
        min = cluster_center.values.min()
        cluster_info = pd.concat([cluster_center, cluster_count], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
        cluster_info.columns = list(df.columns) + [u'类别数目']  # 重命名表头
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, polar=True)
        center_num = cluster_info.values
        feature = ["入会时间(L)", "最近乘机间隔(R)", "飞行次数(F)", "飞行里程(M)", "平均折扣率(C)"]
        N = len(feature)
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
        self.finishSignal.emit((SSE, score))


class readThread(QThread):
    finishSignal = pyqtSignal(pd.DataFrame)
    def __init__(self, parent=None):
        super(readThread, self).__init__(parent)
    def run(self):
        df = pd.read_excel('./temp/data_kmeans.xls')
        self.finishSignal.emit(df)
