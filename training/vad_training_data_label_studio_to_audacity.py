import json
import os
import sys


def transform_json_to_csv(json_data, output_dir, allowed_labels):
    for session in json_data:
        annotations = session.get('annotations', [])
        audio_file = session.get('data', {}).get('audio', '')
        csv_filename = os.path.join(output_dir, os.path.basename(audio_file)+ '.voice_label')
        id = session.get('id')

        csv_rows = []
        # Write CSV content to file
        with open(csv_filename, 'w') as f:
            for annotation in annotations:
                for result in annotation.get('result', []):
                    value = result.get('value', {})
                    labels = value.get('labels', [])
                    start = value.get('start', '')
                    end = value.get('end', '')

                    if len(labels)>1:
                        print(f"skipping segment {start}-{end} with multiple labels {labels} in {audio_file} id={id}")
                        continue
                    label = labels[0]
                    if label not in allowed_labels:
                        print(f"skipping segment {start}-{end} with label {label} in {audio_file} id={id}")
                        continue

                    csv_row = f"{start}\t{end}\t{label}"
                    f.write(csv_row+'\n')

def transform_json_to_csv_utterances(json_data, output_dir):
    for session in json_data:
        annotations = session.get('annotations', [])
        audio_file = session.get('data', {}).get('audio', '')
        id = session.get('id')
        csv_filename = os.path.join(output_dir, os.path.basename(audio_file)+ '.voice_label')

        csv_rows = []
        # Write CSV content to file
        with open(csv_filename, 'w') as f:
            for annotation in annotations:
                utterances = []
                for result in annotation.get('result', []):
                    value = result.get('value', {})
                    labels = value.get('labels', [])
                    start = value.get('start', '')
                    end = value.get('end', '')

                    if len(labels)>1:
                        print(f"skipping segment {start}-{end} with multiple labels {labels} in {audio_file} id={id}")
                        continue
                    label = labels[0]
                    if label!='utterance':
                        continue
                    utterances.append((start,end,'voice'))

                    speech_utterances = []
                    for result in annotation.get('result', []):
                        value = result.get('value', {})
                        labels = value.get('labels', [])
                        start = value.get('start', '')
                        end = value.get('end', '')

                        if len(labels) > 1:
                            print(f"skipping segment {start}-{end} with multiple labels {labels} in {audio_file} id={id}")
                            continue
                        label = labels[0]
                        if label != 'speech':
                            continue

                        utterance_overlap = False
                        for utterance in utterances:
                            #skip speech segments that have overlap with an utterance
                            if (start>=utterance[0] and start<=utterance[1]) or (end>=utterance[0] and end<=utterance[1]):
                                utterance_overlap = True
                                print(f"skipping speech segment {start}-{end} due to overlap with utterance {utterance} in {audio_file} id={id}")
                                break
                        if not utterance_overlap:
                            speech_utterances.append((start, end, 'voice'))

                utterances = utterances + speech_utterances
                for utterance in utterances:
                    csv_row = f"{utterance[0]}\t{utterance[1]}\t{utterance[2]}"
                    f.write(csv_row+'\n')

# Load JSON data from file
label_studio_json_export = sys.argv[1]
output_dir = os.path.join(os.path.dirname(label_studio_json_export),'voice_label')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(label_studio_json_export) as f:
    json_data = json.load(f)

transform_json_to_csv(json_data,output_dir, ['noise','speech'])
output_dir = os.path.join(os.path.dirname(label_studio_json_export),'voice_label_utterances')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

transform_json_to_csv_utterances(json_data, output_dir)

