import sys
from FeatureSelection_func import FeatureSelection
from Kmeans_func import Kmeans
from predict_func import Predict
from PyQt5.QtWidgets import QApplication, QWidget
import os

if __name__ == '__main__':

    if not os.path.exists('./temp'):
        os.makedirs('./temp')  # 创建临时存储的文件夹

    # 实例化
    app = QApplication(sys.argv)

    # 特征选择界面
    ui = FeatureSelection()
    ui1 = Kmeans()
    ui2 = Predict()
    ui.show()

    # kmeans界面

    ui.PB_Kmeans.clicked.connect(ui1.show)
    ui.PB_Kmeans.clicked.connect(ui.close)
    ui.PB_predict.clicked.connect(ui2.show)
    ui.PB_predict.clicked.connect(ui.close)
    # ui.PB_home.clicked.connect(ui.show)
    # ui.PB_home.clicked.connect(ui.close)

    # ui1.PB_Kmeans.clicked.connect(ui1.show)
    # ui1.PB_Kmeans.clicked.connect(ui1.close)
    ui1.PB_predict.clicked.connect(ui2.show)
    ui1.PB_predict.clicked.connect(ui1.close)
    ui1.PB_back.clicked.connect(ui.show)
    ui1.PB_back.clicked.connect(ui1.close)

    ui2.PB_Kmeans.clicked.connect(ui1.show)
    ui2.PB_Kmeans.clicked.connect(ui2.close)
    # ui2.PB_predict.clicked.connect(ui2.show)
    # ui2.PB_predict.clicked.connect(ui2.close)
    ui2.PB_back.clicked.connect(ui.show)
    ui2.PB_back.clicked.connect(ui2.close)

    # 进入程序的主循环，并通过exit函数确保主循环安全结束
    sys.exit(app.exec_())
