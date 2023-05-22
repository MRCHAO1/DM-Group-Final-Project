# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'K-means.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_kmeans(object):
    def setupUi(self, kmeans):
        kmeans.setObjectName("kmeans")
        kmeans.resize(800, 600)
        self.gridLayout = QtWidgets.QGridLayout(kmeans)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_5 = QtWidgets.QLabel(kmeans)
        self.label_5.setWordWrap(True)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.PB_back = QtWidgets.QPushButton(kmeans)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setBold(False)
        font.setWeight(50)
        self.PB_back.setFont(font)
        self.PB_back.setObjectName("PB_back")
        self.verticalLayout.addWidget(self.PB_back)
        self.PB_Kmeans = QtWidgets.QPushButton(kmeans)
        self.PB_Kmeans.setObjectName("PB_Kmeans")
        self.verticalLayout.addWidget(self.PB_Kmeans)
        self.PB_predict = QtWidgets.QPushButton(kmeans)
        self.PB_predict.setObjectName("PB_predict")
        self.verticalLayout.addWidget(self.PB_predict)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(kmeans)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.PB_openfile = QtWidgets.QPushButton(kmeans)
        self.PB_openfile.setObjectName("PB_openfile")
        self.verticalLayout_2.addWidget(self.PB_openfile)
        self.pandasTableWidget = DataTableWidget(kmeans)
        self.pandasTableWidget.setObjectName("pandasTableWidget")
        self.verticalLayout_2.addWidget(self.pandasTableWidget)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(kmeans)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.CB_k = QtWidgets.QComboBox(kmeans)
        font = QtGui.QFont()
        font.setFamily("AcadEref")
        font.setPointSize(9)
        self.CB_k.setFont(font)
        self.CB_k.setObjectName("CB_k")
        self.CB_k.addItem("")
        self.CB_k.addItem("")
        self.CB_k.addItem("")
        self.CB_k.addItem("")
        self.CB_k.addItem("")
        self.CB_k.addItem("")
        self.CB_k.addItem("")
        self.horizontalLayout_2.addWidget(self.CB_k)
        self.PB_confirm = QtWidgets.QPushButton(kmeans)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.PB_confirm.setFont(font)
        self.PB_confirm.setObjectName("PB_confirm")
        self.horizontalLayout_2.addWidget(self.PB_confirm)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(kmeans)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.LE_SSE = QtWidgets.QLineEdit(kmeans)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.LE_SSE.setFont(font)
        self.LE_SSE.setObjectName("LE_SSE")
        self.horizontalLayout.addWidget(self.LE_SSE)
        self.label_3 = QtWidgets.QLabel(kmeans)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.LE_Cont = QtWidgets.QLineEdit(kmeans)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        self.LE_Cont.setFont(font)
        self.LE_Cont.setObjectName("LE_Cont")
        self.horizontalLayout.addWidget(self.LE_Cont)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.LB_radar = QtWidgets.QLabel(kmeans)
        self.LB_radar.setAutoFillBackground(False)
        self.LB_radar.setText("")
        self.LB_radar.setScaledContents(True)
        self.LB_radar.setObjectName("LB_radar")
        self.verticalLayout_2.addWidget(self.LB_radar)
        self.LB_stat = QtWidgets.QLabel(kmeans)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.LB_stat.setFont(font)
        self.LB_stat.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.LB_stat.setAlignment(QtCore.Qt.AlignCenter)
        self.LB_stat.setObjectName("LB_stat")
        self.verticalLayout_2.addWidget(self.LB_stat)
        self.verticalLayout_2.setStretch(2, 3)
        self.verticalLayout_2.setStretch(5, 3)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)

        self.retranslateUi(kmeans)
        QtCore.QMetaObject.connectSlotsByName(kmeans)

    def retranslateUi(self, kmeans):
        _translate = QtCore.QCoreApplication.translate
        kmeans.setWindowTitle(_translate("kmeans", "聚类分析"))
        self.label_5.setText(_translate("kmeans", "本页面根据航空公司客户特征进行客户群体聚类"))
        self.PB_back.setText(_translate("kmeans", "主界面"))
        self.PB_Kmeans.setText(_translate("kmeans", "聚类分析"))
        self.PB_predict.setText(_translate("kmeans", "客户流失预测"))
        self.label.setText(_translate("kmeans", "步骤一导入上一步的特征选择数据（自动为temp/data_kmeans.xls）"))
        self.PB_openfile.setText(_translate("kmeans", "导入数据"))
        self.label_4.setText(_translate("kmeans", "步骤二请输入Kmeans聚类分析的类别数："))
        self.CB_k.setItemText(0, _translate("kmeans", "3"))
        self.CB_k.setItemText(1, _translate("kmeans", "4"))
        self.CB_k.setItemText(2, _translate("kmeans", "5"))
        self.CB_k.setItemText(3, _translate("kmeans", "6"))
        self.CB_k.setItemText(4, _translate("kmeans", "7"))
        self.CB_k.setItemText(5, _translate("kmeans", "8"))
        self.CB_k.setItemText(6, _translate("kmeans", "9"))
        self.PB_confirm.setText(_translate("kmeans", "确认"))
        self.label_2.setText(_translate("kmeans", "误差平方和："))
        self.label_3.setText(_translate("kmeans", "轮廓系数："))
        self.LB_stat.setText(_translate("kmeans", "STATE"))
from qtpandas.views.DataTableView import DataTableWidget