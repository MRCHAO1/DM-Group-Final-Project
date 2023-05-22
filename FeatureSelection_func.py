# D:\Users\Lenovo\anaconda3\envs\chatbot\Scripts\pyuic5.exe FeatureSelection.ui -o FeatureSelection_ui.py
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox

from FeatureSelection_ui import Ui_FeatureSelection
from Kmeans_func import Kmeans
from qtpandas.models.DataFrameModel import DataFrameModel
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

class FeatureSelection(QWidget, Ui_FeatureSelection):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.PB_openfile.clicked.connect(self.msg)
        self.PB_Confirm.clicked.connect(self.featureSelection)
        self.PB_LRFMC.clicked.connect(self.LRMFC)
        self.PB_clean.clicked.connect(self.clean)
        self.model = DataFrameModel()
        self.pandasTableWidget.setViewModel(self.model)

    def PB_set(self, bool):
        self.PB_clean.setEnabled(bool)
        self.PB_openfile.setEnabled(bool)
        self.PB_Kmeans.setEnabled(bool)
        self.PB_LRFMC.setEnabled(bool)
        self.PB_Confirm.setEnabled(bool)
        self.PB_predict.setEnabled(bool)
        self.PB_home.setEnabled(bool)


    def msg(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", "./","All Files (*);;XLS File(.xls)")
        if len(fileName) > 0:
            self.tableShow(fileName)

    def tableShow(self, fileName):
        self.LB_stat.setText("读取数据中")
        self.PB_set(False)
        self.tableShowThread = tableShowThread(fileName)
        self.tableShowThread.finishSignal.connect(self.tableShow_change)
        self.tableShowThread.start()

    def tableShow_change(self, df):
        self.LB_stat.setText("读取数据成功")
        self.PB_set(True)
        self.df_original = df.copy()  # 备份原始数据
        self.model.setDataFrame(self.df_original.head(40))


    def featureSelection(self):
        feature = self.TE_featureSelection.toPlainText().split(' ')
        if len(feature) == 5:
            self.PB_set(False)
            self.LB_stat.setText("特征选择中")
            self.featureSelectionThread = featureSelectionThread(self.df_original, feature)
            self.featureSelectionThread.finishSignal.connect(self.featureSelection_change)
            self.featureSelectionThread.start()
        else:
            QMessageBox(QMessageBox.Critical, '错误', '输入特征格式有误')

    def featureSelection_change(self):
        self.LB_stat.setText("特征选择成功，可进行聚类分析")
        self.PB_set(True)

    def LRMFC(self):
        self.LB_stat.setText("特征选择中")
        self.PB_set(False)
        self.LRMFCThread = LRMFCThread()
        self.LRMFCThread.finishSignal.connect(self.LRMFC_change)
        self.LRMFCThread.start()

    def LRMFC_change(self):
        self.LB_stat.setText("特征选择成功，可进行KMeans分析")
        self.PB_set(True)

    def clean(self):
        self.LB_stat.setText("数据清洗中")
        self.PB_set(False)
        self.cleanThread = cleanThread()
        self.cleanThread.finishSignal.connect(self.clean_change)
        self.cleanThread.start()

    def clean_change(self):
        self.LB_stat.setText("数据清洗成功")
        self.PB_set(True)


class tableShowThread(QThread):
    finishSignal = pyqtSignal(pd.DataFrame)

    def __init__(self, filename, parent=None):
        super(tableShowThread, self).__init__(parent)
        self.filename = filename

    def run(self):
        df = pd.read_excel(self.filename)
        self.finishSignal.emit(df)


class featureSelectionThread(QThread):
    finishSignal = pyqtSignal()

    def __init__(self, df, feature, parent=None):
        super(featureSelectionThread, self).__init__(parent)
        self.df = df
        self.feature = feature

    def run(self):
        self.df[self.feature].to_excel('./temp/data_kmeans.xls', index=None)
        self.finishSignal.emit()

class LRMFCThread(QThread):
    finishSignal = pyqtSignal()
    def __init__(self, parent=None):
        super(LRMFCThread, self).__init__(parent)

    def run(self):
        df_tmp = pd.read_excel('./temp/data_LRFMC.xls')
        df_tmp.to_excel('./temp/data_kmeans.xls', index=None)
        self.finishSignal.emit()
    # 功能代码这个文件内进行添加

class cleanThread(QThread):
    finishSignal = pyqtSignal()
    def __init__(self, parent=None):
        super(cleanThread, self).__init__(parent)

    def run(self):
        df_tmp = pd.read_excel('./temp/data_init.xls')
        df_tmp = df_tmp.loc[df_tmp['FIRST_FLIGHT_DATE'].notnull() &
                        df_tmp['GENDER'].notnull(), :]
        df_tmp.WORK_CITY.fillna(df_tmp.WORK_CITY.mode()[0], inplace=True)
        df_tmp.WORK_PROVINCE.fillna(df_tmp.WORK_PROVINCE.mode()[0], inplace=True)
        df_tmp.WORK_COUNTRY.fillna(df_tmp.WORK_COUNTRY.mode()[0], inplace=True)
        df_tmp['age'].fillna(df_tmp['age'].median(), inplace=True)
        df_tmp = df_tmp.loc[df_tmp['EXPENSE_SUM_YR_1'].notnull() &
                        df_tmp['EXPENSE_SUM_YR_2'].notnull(), :]
        index1 = df_tmp['EXPENSE_SUM_YR_1'] != 0
        index2 = df_tmp['EXPENSE_SUM_YR_2'] != 0
        index3 = (df_tmp['avg_discount'] == 0) & (df_tmp['SEG_KM_SUM'] == 0)
        df_tmp = df_tmp[index1 | index2 | index3]
        df_tmp.to_excel('./temp/data_cleaned.xls', index=None)
        self.finishSignal.emit()

