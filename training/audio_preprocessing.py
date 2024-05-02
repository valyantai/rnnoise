import random

import pandas as pd
import os
from pydub import AudioSegment
import argparse
import csv


def write_index_to_csv(csv_path, index):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['start', 'end', 'source_file'])  # Write header
        writer.writerows(index)

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", "--input_dir",
                                    required=True, help="Input directory")

    parser.add_argument("-l", "--label_dir",
                                    required=False, help="Label directory, set to input dir by default")

    parser.add_argument("-p", "--previous_wav_dir",
                                     required=False, help="Directory with previously preprocessed audio")
    parser.add_argument("-b", "--balance_dataset", type=bool, default=False, help="Should the algorithm remove random " \
                                                                          "noise segments to balance the dataset")

    return parser.parse_args()


def main():
    args = parse_arguments()
    base_path = args.input_dir
    label_path = args.input_dir
    if args.label_dir:
        label_path=args.label_dir
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

    voice_index = []
    noise_index = []
    for file in os.listdir(label_path):
        if not file.endswith('.voice_label'):
            continue
        print(file)
        file_split = file.split('.')
        ext = file_split.pop(-1)
        file_start = '.'.join(file_split)
        try:
            df = pd.read_csv(f"{label_path}/{file}", sep='\t', header=None)
        except:
            print(f"could not read {label_path}/{file}")
            continue
        df.columns = ['start', 'end', 'label']
        audio = AudioSegment.from_file(f"{output_dir}/{file_start}.wav")

        df_sorted = df.sort_values(by='start')
        df_sorted.reset_index(drop=True, inplace=True)

        def filter_rows(row):
            if row['label'] == 'noise':
                # Check if any 'voice' row's range contains the current 'noise' row's range
                return not any((voice_row['start'] <= row['start'] <= voice_row['end']) or
                               (voice_row['start'] <= row['end'] <= voice_row['end'])
                               for _, voice_row in df_sorted.iterrows() if voice_row['label'] == 'voice')
            else:
                return True

        df_sorted_filtered = df_sorted[df_sorted.apply(filter_rows, axis=1)]
        df_sorted_filtered.reset_index(drop=True, inplace=True)

        for i in df_sorted_filtered.index[::-1]:
            row = df_sorted_filtered.iloc[i]
            # print(row)
            start = row['start'] * 1000
            end = row['end'] * 1000
            noise.append((audio[end:], file))
            if row['label'] == 'noise':
                priority_noise.append((audio[start:end], file))
            elif row['label'] != 'emp':
                voice.append((audio[start:end], file))
            audio = audio[:start]
        noise.append((audio, file))

    voice_audio = AudioSegment.empty()
    noise_audio = AudioSegment.empty()
    for voice_segment in voice:
        start= voice_audio.duration_seconds
        voice_audio += voice_segment[0]
        end = voice_audio.duration_seconds
        voice_index.append((start, end, voice_segment[1]))

    for noise_segment in priority_noise:
        start = noise_audio.duration_seconds
        noise_audio += noise_segment[0]
        end = noise_audio.duration_seconds
        noise_index.append((start, end, noise_segment[1]))
        if args.balance_dataset and noise_audio.duration_seconds > voice_audio.duration_seconds:
            break

    # only add non-priority noise if we either don't care about balanced dataset
    # or priority noise is not long enough yet
    if not args.balance_dataset or noise_audio.duration_seconds < voice_audio.duration_seconds:
        random.shuffle(noise)
        for noise_segment in noise:
            start = noise_audio.duration_seconds
            # only take max 5 seconds of a noise segment to avoid using mainly a few long noise profiles and ignoring others
            if noise_segment[0].duration_seconds > 5:
                noise_audio += noise_segment[0][:5000]
            else:
                noise_audio += noise_segment[0]
            end = noise_audio.duration_seconds
            noise_index.append((start, end, noise_segment[1]))
            if args.balance_dataset and noise_audio.duration_seconds > voice_audio.duration_seconds:
                break

    voice_audio = voice_audio.set_frame_rate(48000)
    noise_audio = noise_audio.set_frame_rate(48000)

    output_dir = os.path.join(base_path, 'rnn_wavs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    voice_audio.export(f'{output_dir}/voice.wav', format="wav")
    noise_audio.export(f'{output_dir}/noise.wav', format="wav")

    write_index_to_csv(f'{output_dir}/voice_index.csv', voice_index)
    write_index_to_csv(f'{output_dir}/noise_index.csv', noise_index)

if __name__ == '__main__':
    main()