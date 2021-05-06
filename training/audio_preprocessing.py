import pandas as pd
import os
from pydub import AudioSegment
import sys

base_path = sys.argv[1]
for file in os.listdir(base_path):
    if not file.endswith('.opus') or os.path.exists(f'{base_path}/session_wavs/{file}.wav'):
        continue
    os.system(f'ffmpeg -i {base_path}/{file} -map_channel 0.0.0 {base_path}/session_wavs/{file}.wav')

noise = []
voice = []

noise_previous = AudioSegment.from_file(f"{sys.argv[2]}/noise.wav")
noise.append(noise_previous)
voice_previous = AudioSegment.from_file(f"{sys.argv[2]}/voice.wav")
voice.append(voice_previous)

for file in os.listdir(base_path):
    if not file.endswith('.voice_label'):
        continue
    print(file)
    file_split = file.split('.')
    ext = file_split.pop(-1)
    file_start = '.'.join(file_split)
    try:
        df = pd.read_csv(f"{base_path}/{file}", sep='\t', header=None)
    except:
        print(f"could not read {base_path}/{file}")
        continue
    df.columns = ['start', 'end', 'label']
    audio = AudioSegment.from_file(f"{base_path}/session_wavs/{file_start}.wav")

    for i in df.index[::-1]:
        row = df.iloc[i]
        start = row['start'] * 1000
        end = row['end'] * 1000
        noise.append(audio[end:])
        voice.append(audio[start:end])
        audio = audio[:start]
    noise.append(audio)

voice_audio = AudioSegment.empty()
noise_audio = AudioSegment.empty()
for voice_segment in voice:
    voice_audio += voice_segment

for noise_segment in noise:
    noise_audio += noise_segment

voice_audio = voice_audio.set_frame_rate(48000)
noise_audio = noise_audio.set_frame_rate(48000)
voice_audio.export(f'{base_path}/rnn_wavs/voice.wav', format="wav")
noise_audio.export(f'{base_path}/rnn_wavs/noise.wav', format="wav")