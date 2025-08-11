from typing import Dict


'''
Source:
{
    "id": "reverb1-noise-8k-imda-2021-02942-channel001m-0054272-0055369-2610287",
    "start": 0,
    "duration": 10.97,
    "channel": 0,
    "supervisions": [
        {
            "id": "reverb1-noise-8k-imda-2021-02942-channel001m-0054272-0055369",
            "recording_id": "reverb1-noise-8k-imda-2021-02942-channel001m-0054272-0055369",
            "start": 0.0,
            "duration": 10.97,
            "channel": 0,
            "text": "NOT AS A RICH AND THE ABILITY TO AFFORD UH MAYBE SOME OF UH SOME NEW SUSTAINABLE REUSABLE UH ENERGY MIGHT BE LIMITED YOU KNOW BECAUSE OF UH",
            "language": "English",
            "speaker": "reverb1-noise-8k-imda-2021-02942-channel001m-0054272-0055369"
        }
    ],
    "recording": {
        "id": "reverb1-noise-8k-imda-2021-02942-channel001m-0054272-0055369",
        "sources": [
            {
                "type": "file",
                "channels": [0],
                "source": "/data/projects/71001002/pengyizh/dataset/English-Indon-ASR/Indo_audio_yuhang/home/yuhang001/w2024/k2/icefall/egs/librispeech/ASR/data_12_6//raw_eng_id_reverb/produced_wav/reverb1-noise-8k-imda-2021-02942-channel001m-0054272-0055369.wav"
            }
        ],
        "sampling_rate": 16000,
        "num_samples": 175520,
        "duration": 10.97,
        "channel_ids": [0]
    },
    "type": "MonoCut"}

Target: SLAM
'''

def transform_yizhou_to_slam(data_dict: Dict, task: str = "ASR") -> Dict:
    item = {
        "key": data_dict['key'],
        "language": data_dict['language'],
        "task": "ASR",
        "target": data_dict['target'],
        "path": data_dict["source"],
    }
    if "id" in data_dict:
        item["id"] = data_dict["id"]
    if "duration" in data_dict:
        item["duration"] = data_dict["duration"]
    if "speaker" in data_dict:
        item["speaker"] = data_dict["speaker"]
    if "sampling_rate" in data_dict:
        item["sampling_rate"] = data_dict["sampling_rate"]
    return item

def transform_data_dict(data_dict: Dict) -> Dict:
    try:
        return transform_yizhou_to_slam(data_dict=data_dict)
    except Exception as e:
        print(f"{data_dict} failed")
        raise e
