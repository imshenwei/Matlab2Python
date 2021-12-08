'''音频录制  可以2通道使用

通过pyaudio录制音频
'''
# ————————————————
# 版权声明：本文为CSDN博主「shu_rin」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/shu_rin/article/details/82762812
#
import pyaudio
import wave
import os
import sys


def save_wav_2and8():
    CHUNK = 512

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 0.02
    WAVE_OUTPUT_FILENAME = r"C:\Users\Admin\Desktop\桌面整理\资料\Gitee备份\computer-backup\XQ3.1\专业\实验\实验3 matlab2python\Matlab2Python\修改代码\resources\output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("done")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


save_wav_2and8()
