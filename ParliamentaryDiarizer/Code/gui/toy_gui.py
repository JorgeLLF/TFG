# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_prueba.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import time


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cambiosPlanoButton = QtWidgets.QPushButton(self.centralwidget)
        self.cambiosPlanoButton.setEnabled(False)
        self.cambiosPlanoButton.setGeometry(QtCore.QRect(30, 110, 191, 51))
        self.cambiosPlanoButton.setAutoDefault(False)
        self.cambiosPlanoButton.setDefault(False)
        self.cambiosPlanoButton.setObjectName("cambiosPlanoButton")
        self.clasifPlanosButton = QtWidgets.QPushButton(self.centralwidget)
        self.clasifPlanosButton.setEnabled(False)
        self.clasifPlanosButton.setGeometry(QtCore.QRect(30, 170, 191, 51))
        self.clasifPlanosButton.setObjectName("clasifPlanosButton")
        self.reconFacialButton = QtWidgets.QPushButton(self.centralwidget)
        self.reconFacialButton.setEnabled(False)
        self.reconFacialButton.setGeometry(QtCore.QRect(30, 230, 191, 51))
        self.reconFacialButton.setObjectName("reconFacialButton")
        self.reconVocalButton = QtWidgets.QPushButton(self.centralwidget)
        self.reconVocalButton.setEnabled(False)
        self.reconVocalButton.setGeometry(QtCore.QRect(30, 290, 191, 51))
        self.reconVocalButton.setObjectName("reconVocalButton")
        self.titleLabel = QtWidgets.QLabel(self.centralwidget)
        self.titleLabel.setGeometry(QtCore.QRect(20, 10, 751, 91))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(16)
        self.titleLabel.setFont(font)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.titleLabel.setObjectName("titleLabel")
        self.genFicheroResultButton = QtWidgets.QPushButton(self.centralwidget)
        self.genFicheroResultButton.setEnabled(False)
        self.genFicheroResultButton.setGeometry(QtCore.QRect(30, 410, 191, 51))
        self.genFicheroResultButton.setObjectName("genFicheroResultButton")
        self.preprocessDiarizButton = QtWidgets.QPushButton(self.centralwidget)
        self.preprocessDiarizButton.setEnabled(False)
        self.preprocessDiarizButton.setGeometry(QtCore.QRect(30, 350, 191, 51))
        self.preprocessDiarizButton.setObjectName("preprocessDiarizButton")
        self.labelEstadoProceso = QtWidgets.QLabel(self.centralwidget)
        self.labelEstadoProceso.setGeometry(QtCore.QRect(350, 240, 110, 32))
        self.labelEstadoProceso.setText("")
        self.labelEstadoProceso.setObjectName("labelEstadoProceso")
        font = self.labelEstadoProceso.font()
        font.setPointSize(16)
        self.labelEstadoProceso.setFont(font)
        self.videoSeleccionadoLabel = QtWidgets.QLabel(self.centralwidget)
        self.videoSeleccionadoLabel.setGeometry(QtCore.QRect(62, 86, 55, 16))
        self.videoSeleccionadoLabel.setText("")
        self.videoSeleccionadoLabel.setObjectName("videoSeleccionadoLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.menuArchivo = QtWidgets.QMenu(self.menubar)
        self.menuArchivo.setObjectName("menuArchivo")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSeleccionar = QtWidgets.QAction(MainWindow)
        self.actionSeleccionar.setObjectName("actionSeleccionar")
        self.menuArchivo.addAction(self.actionSeleccionar)
        self.menuArchivo.addSeparator()
        self.menubar.addAction(self.menuArchivo.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.numero = 1
        self.cambiosPlanoButton.clicked.connect(lambda: self.aumentar_valor(self.cambiosPlanoButton,
                                                                            self.clasifPlanosButton))
        self.clasifPlanosButton.clicked.connect(lambda: self.aumentar_valor(self.clasifPlanosButton,
                                                                            self.reconFacialButton))
        self.reconFacialButton.clicked.connect(lambda: self.aumentar_valor(self.reconFacialButton,
                                                                           self.reconVocalButton))
        self.reconVocalButton.clicked.connect(lambda: self.aumentar_valor(self.reconVocalButton,
                                                                          self.preprocessDiarizButton))
        self.preprocessDiarizButton.clicked.connect(lambda: self.aumentar_valor(self.preprocessDiarizButton,
                                                                                self.genFicheroResultButton))
        self.genFicheroResultButton.clicked.connect(lambda: self.aumentar_valor(self.genFicheroResultButton,
                                                                                None))
        self.actionSeleccionar.triggered.connect(self.seleccionar_videos)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.cambiosPlanoButton.setText(_translate("MainWindow", "Detectar cambios de plano"))
        self.clasifPlanosButton.setText(_translate("MainWindow", "Clasificar planos"))
        self.reconFacialButton.setText(_translate("MainWindow", "Reconocer facialmente"))
        self.reconVocalButton.setText(_translate("MainWindow", "Reconocer vocalmente"))
        self.titleLabel.setText(_translate("MainWindow", "Diarizador de sesiones en el Parlamento de Canarias"))
        self.genFicheroResultButton.setText(_translate("MainWindow", "Generar resultado"))
        self.preprocessDiarizButton.setText(_translate("MainWindow", "Preprocesar diarización"))
        self.menuArchivo.setTitle(_translate("MainWindow", "Archivo"))
        self.actionSeleccionar.setText(_translate("MainWindow", "Seleccionar Video"))


    def aumentar_valor(self, boton_actual, siguiente_boton):

        self.numero += 1
        # self.labelNumero.setText(f"Valor actual: {self.numero}")
        # self.labelNumero.adjustSize()

        boton_actual.setEnabled(False)

        # self.labelEstadoProceso.setText(f"Procesando {boton_actual.text()}...")
        # self.labelEstadoProceso.adjustSize()

        # Activar el siguiente botón si existe
        if siguiente_boton:
            siguiente_boton.setEnabled(True)

        # Cambiar el cursor a un cursor de espera
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # Dormir el número de segundos igual al valor actual de la variable
        time.sleep(self.numero)

        # Restaurar el cursor al estado normal
        QtWidgets.QApplication.restoreOverrideCursor()

        self.labelEstadoProceso.setText(f"¡Listo \n{boton_actual.text()}!")
        self.labelEstadoProceso.adjustSize()


    def seleccionar_videos(self):
        opciones = QtWidgets.QFileDialog.Options()
        # Especificar el directorio inicial aquí
        directorio_inicial = "C:/Users/PC/Desktop/ULPGC/GCID/4o/TFG/Proyecto/ParliamentaryDiarizer/Data/Raw/Videos/ShotBoundaryVideos"
        archivos, _ = QtWidgets.QFileDialog.getOpenFileNames(self.centralwidget,
                                                             "Seleccionar Video",
                                                             directorio_inicial,
                                                             "Videos (*.mp4 *.avi *.mov);;Todos los archivos (*)",
                                                             options=opciones)
        if archivos:
            self.rutas_videos = archivos
            self.videoSeleccionadoLabel.setText(f"¡Videos seleccionados!")
            self.labelEstadoProceso.setText(f"")
            self.labelEstadoProceso.adjustSize()
            self.videoSeleccionadoLabel.adjustSize()
            self.cambiosPlanoButton.setEnabled(True)
            self.clasifPlanosButton.setEnabled(False)
            self.reconFacialButton.setEnabled(False)
            self.reconVocalButton.setEnabled(False)
            self.preprocessDiarizButton.setEnabled(False)
            self.genFicheroResultButton.setEnabled(False)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
