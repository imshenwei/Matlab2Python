'''只能8通道使用'''
import pyaudio
import wave


def save_wav_only8():
    CHUNK = 1024
    RESPEAKER_RATE = 48000  # up to 48k
    RESPEAKER_CHANNELS = 2
    RESPEAKER_WIDTH = 2
    # run get_index.py to get index
    RESPEAKER_INDEX = 2  # refer to input device id
    RECORD_SECONDS = 0.02
    WAVE_OUTPUT_FILENAME = "/home/pi/Desktop/修改代码/resources/output.wav"

    p = pyaudio.PyAudio()  # set up the portaudio system.

    stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX,)

    print("* recording")

    frames = []

    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()    # pause playing/recording
    stream.close()          # terminate the stream
    p.terminate()           # terminate the portaudio session

    '''writing file as .wav'''
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(RESPEAKER_CHANNELS)
    wf.setsampwidth(p.get_sample_size(
        p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# save_wav_only8()
