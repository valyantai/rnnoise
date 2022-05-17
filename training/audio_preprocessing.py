import random

import pandas as pd
import os
from pydub import AudioSegment
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", "--input_dir",
                                    required=True, help="Input directory")

    parser.add_argument("-p", "--previous_wav_dir",
                                     required=False, help="Directory with previously preprocessed audio")
    parser.add_argument("-b", "--balance_dataset", type=bool, default=False, help="Should the algorithm remove random " \
                                                                          "noise segments to balance the dataset")

    return parser.parse_args()


def main():
    args = parse_arguments()
    base_path = args.input_dir
    output_dir = os.path.join(base_path, 'session_wavs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in os.listdir(base_path):
        if not (file.endswith('.opus') \
                or file.endswith('.wav'))\
                or os.path.exists(f'{output_dir}/{file}.wav'):
            continue
        os.system(f'ffmpeg -i {base_path}/{file} -map_channel 0.0.0 {output_dir}/{file}.wav')

    noise = []
    priority_noise = []
    voice = []

    if args.previous_wav_dir:
        noise_previous = AudioSegment.from_file(f"{args.previous_wav_dir}/noise.wav")
        noise.append(noise_previous)
        voice_previous = AudioSegment.from_file(f"{args.previous_wav_dir}/voice.wav")
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
        audio = AudioSegment.from_file(f"{output_dir}/{file_start}.wav")

        for i in df.index[::-1]:
            row = df.iloc[i]
            start = row['start'] * 1000
            end = row['end'] * 1000
            noise.append(audio[end:])
            if row['label'] == 'noise':
                priority_noise.append(audio[start:end])
            else:
                voice.append(audio[start:end])
            audio = audio[:start]
        noise.append(audio)

    voice_audio = AudioSegment.empty()
    noise_audio = AudioSegment.empty()
    for voice_segment in voice:
        voice_audio += voice_segment

    for noise_segment in priority_noise:
        noise_audio += noise_segment
        if args.balance_dataset and noise_audio.duration_seconds > voice_audio.duration_seconds:
            break

    # only add non-priority noise if we either don't care about balanced dataset
    # or priority noise is not long enough yet
    if not args.balance_dataset or noise_audio.duration_seconds < voice_audio.duration_seconds:
        random.shuffle(noise)
        for noise_segment in noise:
            noise_audio += noise_segment
            if args.balance_dataset and noise_audio.duration_seconds > voice_audio.duration_seconds:
                break

    voice_audio = voice_audio.set_frame_rate(48000)
    noise_audio = noise_audio.set_frame_rate(48000)

    output_dir = os.path.join(base_path, 'rnn_wavs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    voice_audio.export(f'{output_dir}/voice.wav', format="wav")
    noise_audio.export(f'{output_dir}/noise.wav', format="wav")

if __name__ == '__main__':
    main()