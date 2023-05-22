# D:\Users\Lenovo\anaconda3\envs\chatbot\Scripts\pyuic5.exe predict.ui -o predict_ui.py
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from predict_ui import Ui_Predict
from qtpandas.models.DataFrameModel import DataFrameModel
import pandas as pd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
model = None
x_train = None
class Predict(QWidget, Ui_Predict):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.PB_openfile.clicked.connect(self.msg)
        self.PB_train.clicked.connect(self.train)
        self.PB_clean.clicked.connect(self.clean)
        self.PB_confirm.clicked.connect(self.confirm)
        self.model = DataFrameModel()
        self.pandasTableWidget.setViewModel(self.model)

    def PB_set(self, bool):
        self.PB_back.setEnabled(bool)
        self.PB_openfile.setEnabled(bool)
        self.PB_clean.setEnabled(bool)
        self.PB_train.setEnabled(bool)
        self.PB_predict.setEnabled(bool)
        self.PB_Kmeans.setEnabled(bool)
        self.PB_confirm.setEnabled(bool)

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

    def train(self):
        self.LB_stat.setText("训练中")
        self.PB_set(False)
        self.LE_acc.setText('')
        self.trainThread = trainThread(self.CB_model.currentText())
        self.trainThread.finishSignal.connect(self.train_change)
        self.trainThread.start()

    def train_change(self, acc):
        self.LB_stat.setText("训练完毕")
        self.PB_set(True)
        self.LE_acc.setText(acc)

    def clean(self):
        self.LB_stat.setText("数据预处理中")
        self.PB_set(False)
        self.cleanThread = cleanThread()
        self.cleanThread.finishSignal.connect(self.clean_change)
        self.cleanThread.start()

    def clean_change(self):
        self.LB_stat.setText("数据预处理成功")
        self.PB_set(True)

    def confirm(self):
        self.LB_stat.setText("正在预测")
        self.PB_set(False)
        text = self.LE_test.text().split(',')
        if len(text) != 63:
            self.LB_stat.setText("请确认数据特征数量为63")
            return
        if model == None:
            self.LB_stat.setText("请确认是否完成步骤二的模型训练")
            return
        self.confirmThread = confirmThread(text)
        self.confirmThread.finishSignal.connect(self.confirm_change)
        self.confirmThread.start()

    def confirm_change(self, y_pred):
        self.LB_stat.setText("预测结束")
        self.PB_set(True)
        if y_pred == 0:
            self.LE_result.setText("已流失")
        elif y_pred == 1:
            self.LE_result.setText("准流失")
        else:
            self.LE_result.setText("未流失")

class tableShowThread(QThread):
    finishSignal = pyqtSignal(pd.DataFrame)

    def __init__(self, filename, parent=None):
        super(tableShowThread, self).__init__(parent)
        self.filename = filename

    def run(self):
        df = pd.read_excel(self.filename)
        self.finishSignal.emit(df)

class trainThread(QThread):
    finishSignal = pyqtSignal(str)

    def __init__(self, model, parent=None):
        super(trainThread, self).__init__(parent)
        self.model = model

    def run(self):
        global model
        global x_train
        acc = 0
        data = pd.read_excel('./temp/data_predict.xls')
        data['cat'] = data['cat'].astype('int8')
        x = data.loc[:, data.columns != 'cat']
        y = data.loc[:, data.columns == 'cat']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        x_test = (x_test - x_train.mean(axis=0)) / (x_train.std(axis=0))
        x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0))
        if self.model == 'SVM':
            svm_classifier = svm.SVC(C=1000.0, kernel='rbf', decision_function_shape='ovo', gamma=0.001)
            svm_classifier.fit(x_train, y_train)
            model = svm_classifier
            y_pred = svm_classifier.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
        elif self.model == 'LR':
            # param_grid = {'penalty': ['l2'], 'C': [0.01, 0.1, 1]}
            # grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5, scoring='accuracy')
            # grid.fit(x_train, y_train)
            # acc = grid.best_score_
            lr = LogisticRegression(multi_class = "multinomial", solver="newton-cg")
            lr.fit(x_train, y_train)
            model = lr
            y_pred = lr.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
        elif self.model == 'DT':
            # param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': range(5, 10)}
            # grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')
            # grid.fit(x_train, y_train)
            # acc = grid.best_score_
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            model = dt
            y_pred = dt.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
        elif self.model == 'GBDT':
            # param_grid = {'max_depth': [7]}
            # grid = GridSearchCV(GradientBoostingClassifier(n_estimators=80), param_grid=param_grid, cv=5,
            #                     scoring='accuracy')
            # grid.fit(x_train, y_train)
            # acc = grid.best_score_
            gbdt = GradientBoostingClassifier()
            gbdt.fit(x_train, y_train)
            model = gbdt
            y_pred = gbdt.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
        elif self.model == 'RF':
            clf = RandomForestClassifier()
            clf = clf.fit(x_train, y_train)
            model = clf
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
        data.to_excel('./temp/data_result.xls', index=None)
        self.finishSignal.emit(str(acc)[:9])


class cleanThread(QThread):
    finishSignal = pyqtSignal()
    def __init__(self, parent=None):
        super(cleanThread, self).__init__(parent)

    def run(self):
        df_tmp = pd.read_excel('./temp/data_cleaned.xls')
        # 数据标签
        df_tmp['cat'] = None
        for i in range(len(df_tmp)):
            # 第2年里程积分占最近两年积分比例小于50%的客户定义为已流失
            if df_tmp.loc[i, 'Ration_L1Y_BPS'] < 0.2:
                df_tmp.loc[i, 'cat'] = 0  # 0代表已流失
            # 第2年里程积分占最近两年积分比例在[0.2,0.5)之间的客户定义为准流失
            if 0.2 <= df_tmp.loc[i, 'Ration_L1Y_BPS'] < 0.5:
                df_tmp.loc[i, 'cat'] = 1  # 1代表准流失
            # 第2年里程积分占最近两年积分比例大于等于50%的客户定义为未流失
            if df_tmp.loc[i, 'Ration_L1Y_BPS'] >= 0.5:
                df_tmp.loc[i, 'cat'] = 2  # 2代表未流失
        del df_tmp['Ration_L1Y_BPS'], df_tmp['MEMBER_NO'], df_tmp['GENDER'], df_tmp['WORK_CITY'], df_tmp['WORK_PROVINCE'], df_tmp[
            'WORK_COUNTRY'], df_tmp['ELITE_POINTS_SUM_YR_1']
        # input_data = (input_data-input_data.min())/(input_data.max()-input_data.min())
        df_tmp['FFP_TIME'] = df_tmp['LOAD_TIME'] - df_tmp['FFP_DATE']
        df_tmp['FFP_TIME'] = df_tmp['FFP_TIME'].astype('str').apply(lambda x: x[:-5]).astype('int32')
        del df_tmp['FFP_DATE'], df_tmp['FIRST_FLIGHT_DATE'], df_tmp['LOAD_TIME'], df_tmp['LAST_FLIGHT_DATE']
        del df_tmp['TRANSACTION_DATE']
        df_tmp['cat'] = df_tmp['cat'].astype('int')
        df_tmp.to_excel('./temp/data_predict.xls', index=None)
        self.finishSignal.emit()

class confirmThread(QThread):
    finishSignal = pyqtSignal(int)

    def __init__(self, text, parent=None):
        super(confirmThread, self).__init__(parent)
        self.text = text

    def run(self):
        col = ['MEMBER_NO', 'FFP_DATE', 'FIRST_FLIGHT_DATE', 'GENDER', 'FFP_TIER',
               'WORK_CITY', 'WORK_PROVINCE', 'WORK_COUNTRY', 'age', 'LOAD_TIME',
               'FLIGHT_COUNT', 'FLIGHT_COUNT_QTR_1', 'FLIGHT_COUNT_QTR_2',
               'FLIGHT_COUNT_QTR_3', 'FLIGHT_COUNT_QTR_4', 'FLIGHT_COUNT_QTR_5',
               'FLIGHT_COUNT_QTR_6', 'FLIGHT_COUNT_QTR_7', 'FLIGHT_COUNT_QTR_8',
               'FACD_CLASS_COUNT', 'BASE_POINTS_SUM', 'BASE_POINTS_SUM_QTR_1',
               'BASE_POINTS_SUM_QTR_2', 'BASE_POINTS_SUM_QTR_3',
               'BASE_POINTS_SUM_QTR_4', 'BASE_POINTS_SUM_QTR_5',
               'BASE_POINTS_SUM_QTR_6', 'BASE_POINTS_SUM_QTR_7',
               'BASE_POINTS_SUM_QTR_8', 'ELITE_POINTS_SUM_YR_1',
               'ELITE_POINTS_SUM_YR_2', 'EXPENSE_SUM_YR_1', 'EXPENSE_SUM_YR_2',
               'SEG_KM_SUM', 'WEIGHTED_SEG_KM', 'LAST_FLIGHT_DATE', 'AVG_FLIGHT_COUNT',
               'AVG_BASE_POINTS_SUM', 'DAYS_FROM_BEGIN_TO_FIRST',
               'DAYS_FROM_LAST_TO_END', 'AVG_FLIGHT_INTERVAL', 'MAX_FLIGHT_INTERVAL',
               'MILEAGE_IN_COUNT', 'ADD_POINTS_SUM_YR_1', 'ADD_POINTS_SUM_YR_2',
               'EXCHANGE_COUNT', 'TRANSACTION_DATE', 'avg_discount',
               'P1Y_Flight_Count', 'L1Y_Flight_Count', 'P1Y_BASE_POINTS_SUM',
               'L1Y_BASE_POINTS_SUM', 'ELITE_POINTS_SUM', 'ADD_POINTS_SUM',
               'Eli_Add_Point_Sum', 'L1Y_ELi_Add_Points', 'Points_Sum',
               'L1Y_Points_Sum', 'Ration_L1Y_Flight_Count', 'Ration_P1Y_Flight_Count',
               'Ration_P1Y_BPS', 'Ration_L1Y_BPS', 'Point_Chg_NotFlight']
        data = pd.DataFrame({col[i]: [self.text[i]] for i in range(len(col))})
        del data['Ration_L1Y_BPS'], data['MEMBER_NO'], data['GENDER'], data['WORK_CITY'], data['WORK_PROVINCE'], data[
            'WORK_COUNTRY'], data['ELITE_POINTS_SUM_YR_1']
        data['FFP_TIME'] = pd.to_datetime(data['LOAD_TIME']) - pd.to_datetime(data['FFP_DATE'])
        data['FFP_TIME'] = data['FFP_TIME'].astype('str').apply(lambda x: x[:-5]).astype('int32')
        del data['FFP_DATE'], data['FIRST_FLIGHT_DATE'], data['LOAD_TIME'], data['LAST_FLIGHT_DATE']
        del data['TRANSACTION_DATE']
        data[data.columns] = data[data.columns].astype('float64')
        x = (data - x_train.mean(axis=0)) / (x_train.std(axis=0))
        y_pred = model.predict(x)
        self.finishSignal.emit(y_pred[0])
