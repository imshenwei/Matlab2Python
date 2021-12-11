'''未知函数

打开麦克风
'''
from numpy import *
import pyaudio
import numpy as np
import time


class mic_array():
    def __init__(self, RESPEAKER_INDEX=2, CHUNK=1024, RESPEAKER_RATE=48000, RESPEAKER_CHANNELS=8, RESPEAKER_WIDTH=2, RECORD_SECONDS=0.02):
        self.p = pyaudio.PyAudio()
        self.RESPEAKER_INDEX = RESPEAKER_INDEX
        self.CHUNK = CHUNK
        self.RESPEAKER_RATE = RESPEAKER_RATE
        self.RESPEAKER_CHANNELS = RESPEAKER_CHANNELS
        self.RESPEAKER_WIDTH = RESPEAKER_WIDTH
        self.RECORD_SECONDS = RECORD_SECONDS
        self.mic_init()

    def mic_init(self):
        name_string = []
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name_string = self.p.get_device_info_by_host_api_device_index(
                    0, i).get('name')
                if(name_string[0:5] == "seeed"):
                    self.RESPEAKER_INDEX = i
                    print("Find id=", i)

        self.stream = self.p.open(
            rate=self.RESPEAKER_RATE,
            format=self.p.get_format_from_width(self.RESPEAKER_WIDTH),
            channels=self.RESPEAKER_CHANNELS,
            input=True,
            start=False,
            input_device_index=self.RESPEAKER_INDEX,
        )
        time.sleep(0.5)

    def get_data(self, data_index=0):
        raw_data = self.get_raw_data(data_index)
        data = self.TrackAlignment(raw_data)
        return data

    # 录制8通道音频
    def get_raw_data(self, data_index=0):
        frames = []
        self.stream.start_stream()
        if(data_index > 0):
            for i in range(0, data_index+1):
                data = self.stream.read(self.CHUNK)
                frames.append(data)
        else:
            frames = [self.stream.read(self.CHUNK)]
        self.stream.stop_stream()
        data = np.fromstring(frames[data_index], dtype=np.int16)
        decoded = np.reshape(data, [self.CHUNK, 8])
        return decoded

    def TrackAlignment(self, data):
        y = data
        s_x = np.argsort(np.max(y, axis=0))
        y2 = []
        if abs(s_x[0]-s_x[1]) == 1:
            s_x_max = max(s_x[0:1])
            if(s_x_max == 1):
                y2 = y[:, 2:8]
            elif(s_x_max < 7 and s_x_max > 1):
                y2 = np.hstack((y[:, s_x_max+1:8], y[:, 0:s_x_max-1]))
            else:
                y2 = y[:, 0:6]
        else:
            y2 = y[:, 1:7]
        return y2

    def close(self):
        self.stream.close()
        self.p.terminate()
