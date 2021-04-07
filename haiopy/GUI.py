#!/usr/bin/env python
import numpy as np                      # scientific computing lib
import sounddevice as sd                # sounddevice / hostapi handling
import sys                              # used for printing err to std stream
import matplotlib.pyplot as plt         # plots and graphs
import os                               # functions for operatingsystem
import os.path                          # functions for operatingsystem
from PyQt5 import QtWidgets             # GUI framework
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog,
                             QGridLayout, QTabWidget,  QProgressBar,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QWidget, QFormLayout,
                             QFileDialog, QSizePolicy)

import pyqtgraph as pg                  # scientific plots
from pyqtgraph.Qt import QtGui, QtCore  # GUI framework
import pyaudio                          # cross-platform audio I/O
import struct                           # interpret bytes as packed binary data
from scipy.fftpack import fft           # Fast-Fourier-Transform
import scipy.io.wavfile as wavfile      # read .wav-files
import haiopy as haiopy                 # package for audio rec, play, monitor
# ----------------------------------------------
# enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
# use highdpi icons
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# ----------------------------------------------


class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette() 
        styleLabel = QLabel("Gruppe 6 - Audio Devices:")

        settingButton = QPushButton('Settings')
        settingButton.clicked.connect(settingButton_on_click)

        self.RecordBox()
        self.playMusicBox()
        self.createProgressBar()
        # self.SpectrumAnalyzer()

        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addStretch(50)
        topLayout.addWidget(settingButton)

        mainLayout = QGridLayout()
        mainLayout.setSpacing(20)
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.RecordFunction, 1, 0)
        mainLayout.addWidget(self.playMusicFuntion, 1, 1, 1, 1)
        mainLayout.addWidget(self.progressBar, 2, 0, 1, 2)
        mainLayout.setRowStretch(3, 1)
        # mainLayout.addWidget(self.PlotBox, 4, 0, 1, 4)
        self.setLayout(mainLayout)

        self.setWindowTitle("GUI")
        self.setFixedSize(400, 280)
        QApplication.setStyle('Fusion')

    def RecordBox(self):
        self.RecordFunction = QGroupBox("Record")
        self.RecordFunction.setCheckable(True)
        self.RecordFunction.setChecked(True)

        topLayout = QFormLayout()
        self.duration_input = QLineEdit('')
        self.duration_input.setFixedWidth(80)
        topLayout.addRow("Duration/s:", self.duration_input)

        self.RecordButton = QPushButton("Record")
        self.RecordButton.setCheckable(True)
        self.RecordButton.setChecked(False)
        self.RecordButton.clicked.connect(self.RecordButtonClicked)
        # self.RecordButton.clicked.connect(recordButton_on_clicked)
        self.RecordButton.setStyleSheet("background-color : normal")

        self.StopButton = QPushButton("Stop")
        self.StopButton.setDefault(True)
        self.StopButton.clicked.connect(self.StopButtonClicked)

        PlotButton = QPushButton('Plot')
        PlotButton.clicked.connect(PlotButton_on_clicked)

        layout = QGridLayout()
        # layout.setSpacing(5)
        layout.addLayout(topLayout, 0, 0, 1, 2)
        layout.addWidget(self.RecordButton, 1, 0)
        layout.addWidget(self.StopButton, 1, 1)
        layout.addWidget(PlotButton, 2, 0, 1, 2)
        # layout.addStretch(1)
        self.RecordFunction.setLayout(layout)

    def RecordButtonClicked(self):
        if self.RecordButton.isChecked():
            self.RecordButton.setStyleSheet("background-color: red")
        else:
            self.RecordButton.setStyleSheet("background-color: normal")

        self.duration = self.duration_input.text()
        self.duration = int(self.duration)

        self.duration_input.setEnabled(False)
        self.progressBar.setRange(0, self.duration)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advanceProgressBar)
        self.timer.start(1000)

        self.recordsi = haiopy.Record('wav', device_in=0)
        self.recordedsignal = self.recordsi.record(duration=self.duration)

    def StopButtonClicked(self):
        self.RecordButton.setStyleSheet("background-color: normal")
        self.RecordButton.setChecked(False)
        self.duration_input.setEnabled(True)
        self.timer.stop()
        self.recordsi.on_stop()
        self.progressBar.setValue(0)
        self.curVal = 0

    # def getDuration(self):
    #     print(self.duration_input.text())
    #     self.duration_input.setEnabled(False)

    def playMusicBox(self):
        self.playMusicFuntion = QTabWidget()
        self.playMusicFuntion.setSizePolicy(QSizePolicy.Preferred,
                                            QSizePolicy.Ignored)

        tab1 = QWidget()
        self.OpenWav = QPushButton('Open')
        self.OpenWav.clicked.connect(self.openFileNamesDialog)
        self.OpenWav.setFixedWidth(50)

        topLayout = QFormLayout()
        self.show_openedFile = QLineEdit('file path')
        self.show_openedFile.setEnabled(False)
        topLayout.addRow('File Name:', self.show_openedFile)

        self.Play_PlayButton = QPushButton("Play")
        self.Play_PlayButton.setCheckable(True)
        self.Play_PlayButton.setChecked(False)
        self.Play_PlayButton.clicked.connect(self.Play_PlayButtonClicked)
        self.Play_PlayButton.setStyleSheet("background-color : normal")

        self.Play_StopButton = QPushButton("Stop")
        self.Play_StopButton.setDefault(True)
        self.Play_StopButton.clicked.connect(self.Play_StopButtonClicked)

        self.Play_PlotButton = QPushButton('Plot')
        self.Play_PlotButton.clicked.connect(self.plot_wavfile)

        layout = QGridLayout()
        layout.addWidget(self.OpenWav, 0, 0)
        layout.addLayout(topLayout, 1, 0, 1, 2)
        layout.addWidget(self.Play_PlayButton, 2, 0)
        layout.addWidget(self.Play_StopButton, 2, 1)
        layout.addWidget(self.Play_PlotButton, 3, 0, 1, 2)
        layout.setContentsMargins(5, 5, 5, 5)
        tab1.setLayout(layout)

        tab2 = QWidget()
        self.playrec_OpenWav = QPushButton('Open')
        self.playrec_OpenWav.clicked.connect(self.openFileNamesDialog2)
        self.playrec_OpenWav.setFixedWidth(50)

        topLayout_playrec = QFormLayout()
        self.playrec_show_openedFile = QLineEdit('file path')
        self.playrec_show_openedFile.setEnabled(False)
        # self.show_openFile.setFixedWidth(100)
        topLayout_playrec.addRow('File Name:', self.playrec_show_openedFile)

        self.recplay_RecordButton = QPushButton("Record")
        self.recplay_RecordButton.setCheckable(True)
        self.recplay_RecordButton.setChecked(False)
        self.recplay_RecordButton.clicked.connect(self.
                                                  recplay_RecordButtonClicked)
        self.recplay_RecordButton.setStyleSheet("background-color : normal")

        self.playrec_StopButton = QPushButton("Stop")
        self.playrec_StopButton.setDefault(True)
        self.playrec_StopButton.clicked.connect(self.playrec_StopButtonClicked)

        layout_playrec = QGridLayout()
        layout_playrec.addWidget(self.playrec_OpenWav, 0, 0)
        layout_playrec.addLayout(topLayout_playrec, 1, 0, 1, 2)
        layout_playrec.addWidget(self.recplay_RecordButton, 2, 0)
        layout_playrec.addWidget(self.playrec_StopButton, 2, 1)
        # layout.setContentsMargins(5, 5, 5, 5)

        # layout_playrec.setContentsMargins(5, 5, 5, 5)
        tab2.setLayout(layout_playrec)

        self.playMusicFuntion.addTab(tab1, "Play")
        self.playMusicFuntion.addTab(tab2, "PlayRecord")

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.files, _ = QFileDialog.getOpenFileNames(self,
                                                     "QFileDialog.\
                                                         getOpenFileNames()",
                                                     "",
                                                     "All Files (*);;\
                                                         Python Files (*.py)",
                                                     options=options)
        if self.files:
            self.filesName = str(self.files).split('/')
            self.filesName = self.filesName[-1][:-2]
            self.show_openedFile.setText(self.filesName)
            # print(self.files)

    def openFileNamesDialog2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.files, _ = QFileDialog.getOpenFileNames(self,
                                                     "QFileDialog.\
                                                         getOpenFileNames()",
                                                     "",
                                                     "All Files (*);;\
                                                         Python Files (*.py)",
                                                     options=options)
        if self.files:
            self.filesName = str(self.files).split('/')
            self.filesName = self.filesName[-1][:-2]
            self.playrec_show_openedFile.setText(self.filesName)

    def recplay_RecordButtonClicked(self):
        if self.playrec_show_openedFile.text() != 'file path':
            self.suffix = self.filesName.split('.')
            if self.filesName and self.suffix[-1] == 'wav':
                self.recplay_RecordButton.setStyleSheet("background-color:red")
                fi = str(self.files[0])
                self.playrectest = haiopy.PlayRecord(audio_in='wav',
                                                     audio_out=fi,
                                                     device_in=sd.default.device[0],
                                                     device_out=sd.default.device[1],
                                                     sampling_rate=44100)
                self.playrectest.playrec()
                self.progressBar.setRange(0, self.playrectest.duration)

                self.timer = QTimer(self)
                self.timer.timeout.connect(self.advanceProgressBar)
                self.timer.start(1000)

            else:
                self.recplay_RecordButton.setStyleSheet("background-\
                                                        color: normal")
                popup = PopupWindow(self)
                popup.setGeometry(400, 400, 500, 100)
                popup.show()
                self.recplay_RecordButton.setChecked(False)
        else:
            self.recplay_RecordButton.setStyleSheet("background-\
                                                    color: normal")
            popup = PopupWindow(self)
            popup.setGeometry(400, 400, 500, 100)
            popup.show()
            self.recplay_RecordButton.setChecked(False)

    def Play_PlayButtonClicked(self):
        if self.show_openedFile.text() != 'file path':
            self.suffix = self.filesName.split('.')

            if self.filesName and self.suffix[-1] == 'wav':
                self.playsi = haiopy.Play(self.filesName,
                                          device_out=sd.default.device[1])
                self.playsi.play()
                self.progressBar.setRange(0, self.playsi.duration)

                self.timer = QTimer(self)
                self.timer.timeout.connect(self.advanceProgressBar)
                self.timer.start(1000)
            else:
                popup = PopupWindow(self)
                popup.setGeometry(400, 400, 500, 100)
                popup.show()
                self.Play_PlayButton.setChecked(False)
        else:
            popup = PopupWindow(self)
            popup.setGeometry(400, 400, 500, 100)
            popup.show()
            self.Play_PlayButton.setChecked(False)

    def Play_StopButtonClicked(self):
        self.timer.stop()
        self.Play_PlayButton.setStyleSheet("background-color: normal")
        self.Play_PlayButton.setChecked(False)
        self.playsi.output_stream.abort()
        self.progressBar.setValue(0)
        self.curVal = 0

    def playrec_StopButtonClicked(self):
        self.timer.stop()
        self.recplay_RecordButton.setStyleSheet("background-color: normal")
        self.recplay_RecordButton.setChecked(False)
        self.playrectest.playrec_stream.abort()
        self.progressBar.setValue(0)
        self.curVal = 0

    def plot_wavfile(self):
        if self.show_openedFile.text() != 'file path':
            self.suffix = self.filesName.split('.')
            if self.filesName and self.suffix[-1] == 'wav':
                print(self.files[0])
                myAudioFilename = str(self.files[0])
                # homedir -> audiodir -> my wav files
                dataset_path = os.path.join(os.environ['HOME'], 'audio')
                wavedata = os.path.join(dataset_path, myAudioFilename)

                sampleRate, audioBuffer = wavfile.read(wavedata)
                duration = len(audioBuffer)/sampleRate
                # time vector
                time = np.arange(0, duration, 1/sampleRate)
                if audioBuffer.shape[1] == 2:
                    newaudioBuffer = []
                    for i in range(len(audioBuffer)):
                        newaudioBuffer.append(audioBuffer[i][0])
                    newaudioBuffer = np.array(newaudioBuffer)
                    time = time[:len(newaudioBuffer)]
                    plt.plot(time, newaudioBuffer)
                else:
                    time = time[:len(audioBuffer)]
                    plt.plot(time, audioBuffer)
                plt.xlabel('Time [s]')
                plt.ylabel('Amplitude')
                plt.title(myAudioFilename)
                plt.show()

            else:
                popup = PopupWindow(self)
                popup.setGeometry(400, 400, 500, 100)
                popup.show()
                self.Play_PlayButton.setChecked(False)
        else:
            popup = PopupWindow(self)
            popup.setGeometry(400, 400, 500, 100)
            popup.show()
            self.Play_PlayButton.setChecked(False)

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)

    def advanceProgressBar(self):
        self.curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        # print(maxVal)
        self.progressBar.setValue(self.curVal + 1)
        if self.curVal >= maxVal:
            self.timer.stop()
            self.progressBar.setValue(0)
            self.curVal = 0
            if self.curVal == 0:
                self.RecordButton.setStyleSheet("background-color: normal")
                self.recplay_RecordButton.setStyleSheet("background-\
                                                        color:normal")
                self.recplay_RecordButton.setChecked(False)
                self.Play_PlayButton.setChecked(False)
                self.RecordButton.setChecked(False)
                self.duration_input.setEnabled(True)


class PopupWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel('No files has been chosen or its not a wav file\n \n \
                            please chose a wav file', self)
        # self.label.setFont.("font-weight: bold")
        self.setWindowTitle('WARNING')


class realtime_Plot(QDialog):
    def __init__(self):
        super(realtime_Plot, self).__init__()

        self.setWindowTitle('Real Time Input Plot')
        self.resize(800, 500)
        # self.setWindowModality(Qt.ApplicationModal)

        self.traces = dict()
        self.color = (224, 223, 227)
        self.win_1 = pg.GraphicsLayoutWidget()
        self.win_1.setBackground(self.color)
        self.win_2 = pg.GraphicsLayoutWidget()
        self.win_2.setBackground(self.color)

        self.wf_xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        self.wf_xaxis = pg.AxisItem(orientation='bottom')
        self.wf_xaxis.setTicks([self.wf_xlabels])

        self.wf_ylabels = [(0, '0'), (-1, '-1'), (1, '1')]
        self.wf_yaxis = pg.AxisItem(orientation='left')
        self.wf_yaxis.setTicks([self.wf_ylabels])

        self.sp_xlabels = [
            (np.log10(20), '20'), (np.log10(50), '50'), (np.log10(100), '100'),
            (np.log10(250), '250'), (np.log10(500), '500'),
            (np.log10(1000), '1k'), (np.log10(4000), '4k'),
            (np.log10(8000), '8k'), (np.log10(20000), '20k')]
        self.sp_xaxis = pg.AxisItem(orientation='bottom')
        self.sp_xaxis.setTicks([self.sp_xlabels])

        self.sp_ylabels = [(0, '0'), (-1, '-12'), (-2, '-24'), (-3, '-48')]
        self.sp_yaxis = pg.AxisItem(orientation='left')
        self.sp_yaxis.setTicks([self.sp_ylabels])

        self.waveform = self.win_1.addPlot(
            title='Waveform', row=1, col=1, axisItems={'bottom': self.wf_xaxis,
                                                       'left': self.wf_yaxis},)

        self.spectrum = self.win_2.addPlot(
            title='Spectrum', row=2, col=1, axisItems={'bottom': self.sp_xaxis,
                                                       'left': self.sp_yaxis},)

        self.waveform.showGrid(x=True, y=True, alpha=0.3)
        self.waveform.setMouseEnabled(x=False, y=False)
        self.spectrum.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum.setMouseEnabled(x=False, y=False)

        layout = QHBoxLayout()
        layout.addWidget(self.win_1)
        layout.addWidget(self.win_2)

        self.setLayout(layout)

# ------------------------------- just for test ------------------------
# pyaudio should be deleted

        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )

        # waveform and spectrum x points
        self.x = np.arange(0, 2 * self.CHUNK, 2)
        self.f = np.linspace(0, 20000, 1024)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, data_x, data_y):
        if name in self.traces:
            self.traces[name].setData(data_x, data_y)
        else:
            if name == 'waveform':
                self.traces[name] = self.waveform.plot(pen='b', width=4)
                self.waveform.setYRange(-1, 1, padding=0.05)
                self.waveform.setXRange(0, 2 * self.CHUNK, padding=0.05)
            if name == 'spectrum':
                self.traces[name] = self.spectrum.plot(pen='m', width=4)
                self.spectrum.setLogMode(x=True, y=True)
                self.spectrum.setYRange(-4, 0, padding=0.05)
                self.spectrum.setXRange(
                    np.log10(20), np.log10(self.RATE / 2), padding=0.05)

    def update(self):
        self.wf_data = self.stream.read(self.CHUNK)
        self.wf_data = struct.unpack(str(2 * self.CHUNK) + 'B', self.wf_data)
        self.wf_data = np.array(self.wf_data, dtype='b')[::2] + 128
        self.wf_data_ip = np.interp(self.wf_data,
                                    (self.wf_data.min(),
                                     self.wf_data.max()),
                                    (-1, +1))
        self.set_plotdata(name='waveform',
                          data_x=self.x,
                          data_y=self.wf_data_ip,)

        self.sp_data = fft(np.array(self.wf_data, dtype='int8') - 128)
        self.sp_data = np.abs(self.sp_data[0:int(self.CHUNK / 2)]) * 2 / \
                             (128 * self.CHUNK)
        self.set_plotdata(name='spectrum', data_x=self.f, data_y=self.sp_data)

# -----------------------------------------------------------------------------

    def animation(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)
        self.start()
        # self.setLayout(self.layout)
        self.exec_()


class settings(QDialog):
    """Dialog window for choosing sound device."""

    def __init__(self):
        super(settings, self).__init__()

        self.setWindowTitle('Settings')
        # self.b1 = QPushButton("ok", self)
        # self.b1.move(50, 50)
        self.resize(200, 200)
        # self.move(650, 450)
        self.setWindowModality(Qt.ApplicationModal)
        self.layout = QGridLayout()

        # Create central Widget
        self.centralWidget = QWidget(self)

        # Create combobox and add available Host APIs
        self.host_label = QLabel('Host APis:')
        self.host = QComboBox(self.centralWidget)
        self.host_label.setBuddy(self.host)
        self.host.setToolTip('This are the HOST APIs:')
        for hostapi in sd.query_hostapis():
            self.host.addItem(hostapi['name'])
        self.layout.addWidget(self.host_label, 0, 0)
        self.layout.addWidget(self.host, 0, 1)
        self.host.currentTextChanged.connect(self.host_changed)

        # create combobox and add available inputs
        self.inputs_label = QLabel('Input:')
        self.inputs = QComboBox(self.centralWidget)
        self.inputs_label.setBuddy(self.inputs)
        self.inputs.setToolTip('Choose your sound device input channels')
        self.hostapi = sd.query_hostapis(self.host.currentIndex())
        for idx in self.hostapi['devices']:
            if sd.query_devices(idx)['max_input_channels'] > 0:
                self.inputs.addItem(sd.query_devices(idx)['name'])
        self.inputs.currentTextChanged.connect(self.input_changed)
        self.layout.addWidget(self.inputs_label, 1, 0)
        self.layout.addWidget(self.inputs, 1, 1)

        # create combobox and add available outputs
        self.outputs_label = QLabel('Outputs:')
        self.outputs = QComboBox(self.centralWidget)
        self.outputs_label.setBuddy(self.outputs)
        self.outputs.setToolTip('Choose your sound device output channels')
        self.hostapi = sd.query_hostapis(self.host.currentIndex())
        for idx in self.hostapi['devices']:
            if sd.query_devices(idx)['max_output_channels'] > 0:
                self.outputs.addItem(sd.query_devices(idx)['name'])
        self.layout.addWidget(self.outputs_label, 2, 0)
        self.layout.addWidget(self.outputs, 2, 1)
        self.outputs.currentTextChanged.connect(self.output_changed)

        self.setLayout(self.layout)
        self.exec_()

    def get_input_id_by_name(self, channel_name):
        devices_list = sd.query_devices()
        for index, device_msg_dict in enumerate(devices_list):
            if channel_name == device_msg_dict["name"] and \
                               device_msg_dict["max_input_channels"] > 0:
                return index
        else:
            raise ValueError("cannot find the input channel")

    def get_output_id_by_name(self, channel_name):
        devices_list = sd.query_devices()
        for index, device_msg_dict in enumerate(devices_list):
            if channel_name == device_msg_dict["name"] and \
                               device_msg_dict["max_output_channels"] > 0:
                return index
        else:
            raise ValueError("cannot find the output channel")

    def host_changed(self, host):
        # set host comman not found
        print("Sound device(host) is alread changed to xxx")

    def input_changed(self, input_name):
        input_id = self.get_input_id_by_name(input_name)
        sd.default.device[0] = input_id  # index 1 is output, index 0 is input
        print("inputs changed:", input_name)

    def output_changed(self, output_name):
        output_id = self.get_output_id_by_name(output_name)
        sd.default.device[1] = output_id  # index 1 is output, index 0 is input
        print("outputs changed:", output_name)


# @pyqtSlot()
# def recordButton_on_clicked(filesname, device_out, channels_out):
#     playsi = Play(filesname,device_out=device_out, channels_out=channels_out)
#     playsi.play()

@pyqtSlot()
def PlotButton_on_clicked():
    plot = realtime_Plot()
    plot.show()
    plot.animation()


@pyqtSlot()
def settingButton_on_click():
    settings()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    mainGui = WidgetGallery()
    mainGui.show()
    # mainGui.animation()
    sys.exit(app.exec_())
