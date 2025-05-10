from __future__ import annotations

import functools
from typing import List, Tuple

import gradio as gr
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
import sys
import os
import time
import re
from pydub import AudioSegment

base_path = os.path.dirname(sys.path[0])
sys.path.append(os.path.join(base_path, "src"))
print(sys.path)
from seamless_communication.models.inference.translator import Translator

DESCRIPTION = """# SeamlessM4T  with Streaming

[SeamlessM4T](https://github.com/facebookresearch/seamless_communication) is designed to provide high-quality
translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.

This unified model enables multiple tasks like Speech-to-Speech (S2ST), Speech-to-Text (S2TT), Text-to-Speech (T2ST)
translation and more, without relying on multiple separate models.
"""

TASK_NAMES = [
    "S2ST (Speech to Speech translation)",
    "S2TT (Speech to Text translation)",
    "T2ST (Text to Speech translation)",
    "T2TT (Text to Text translation)",
    "ASR (Automatic Speech Recognition)",
]

# Language dict
language_code_to_name = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "arb": "Modern Standard Arabic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "North Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "est": "Estonian",
    "eus": "Basque",
    "fin": "Finnish",
    "fra": "French",
    "gaz": "West Central Oromo",
    "gle": "Irish",
    "glg": "Galician",
    "guj": "Gujarati",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kam": "Kamba",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "kea": "Kabuverdianu",
    "khk": "Halh Mongolian",
    "khm": "Khmer",
    "kir": "Kyrgyz",
    "kor": "Korean",
    "lao": "Lao",
    "lit": "Lithuanian",
    "ltz": "Luxembourgish",
    "lug": "Ganda",
    "luo": "Luo",
    "lvs": "Standard Latvian",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mni": "Meitei",
    "mya": "Burmese",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokm\u00e5l",
    "npi": "Nepali",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Odia",
    "pan": "Punjabi",
    "pbt": "Southern Pashto",
    "pes": "Western Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sna": "Shona",
    "snd": "Sindhi",
    "som": "Somali",
    "spa": "Spanish",
    "srp": "Serbian",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Northern Uzbek",
    "vie": "Vietnamese",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "yue": "Cantonese",
    "zlm": "Colloquial Malay",
    "zsm": "Standard Malay",
    "zul": "Zulu",
}
LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}

# Source langs: S2ST / S2TT / ASR don't need source lang
# T2TT / T2ST use this
text_source_language_codes = [
    "afr",
    "amh",
    "arb",
    "ary",
    "arz",
    "asm",
    "azj",
    "bel",
    "ben",
    "bos",
    "bul",
    "cat",
    "ceb",
    "ces",
    "ckb",
    "cmn",
    "cym",
    "dan",
    "deu",
    "ell",
    "eng",
    "est",
    "eus",
    "fin",
    "fra",
    "gaz",
    "gle",
    "glg",
    "guj",
    "heb",
    "hin",
    "hrv",
    "hun",
    "hye",
    "ibo",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kan",
    "kat",
    "kaz",
    "khk",
    "khm",
    "kir",
    "kor",
    "lao",
    "lit",
    "lug",
    "luo",
    "lvs",
    "mai",
    "mal",
    "mar",
    "mkd",
    "mlt",
    "mni",
    "mya",
    "nld",
    "nno",
    "nob",
    "npi",
    "nya",
    "ory",
    "pan",
    "pbt",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "slv",
    "sna",
    "snd",
    "som",
    "spa",
    "srp",
    "swe",
    "swh",
    "tam",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
    "yor",
    "yue",
    "zsm",
    "zul",
]
TEXT_SOURCE_LANGUAGE_NAMES = sorted(
    [language_code_to_name[code] for code in text_source_language_codes]
)

# Target langs:
# S2ST / T2ST
s2st_target_language_codes = [
    "eng",
    "arb",
    "ben",
    "cat",
    "ces",
    "cmn",
    "cym",
    "dan",
    "deu",
    "est",
    "fin",
    "fra",
    "hin",
    "ind",
    "ita",
    "jpn",
    "kor",
    "mlt",
    "nld",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "spa",
    "swe",
    "swh",
    "tel",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
]
S2ST_TARGET_LANGUAGE_NAMES = sorted(
    [language_code_to_name[code] for code in s2st_target_language_codes]
)
# S2TT / ASR
S2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
# T2TT
T2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES

# Download sample input audio files
filenames = ["assets/sample_input.mp3", "assets/sample_input_2.mp3"]

for filename in filenames:
    hf_hub_download(
        repo_id="facebook/seamless_m4t",
        repo_type="space",
        filename=filename,
        local_dir=".",
    )

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 120  # in seconds
DEFAULT_TARGET_LANGUAGE = "French"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

translator = Translator(
    model_name_or_card="seamlessM4T_large",
    vocoder_name_or_card="vocoder_36langs",
    device=device,
    dtype=torch.float16 if "cuda" in device.type else torch.float32,
)


def predict(
        task_name: str,
        audio_source: str,
        input_audio_mic: str | None,
        input_audio_file: str | None,
        input_text: str | None,
        source_language: str | None,
        target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    task_name = task_name.split()[0]
    source_language_code = (
        LANGUAGE_NAME_TO_CODE[source_language] if source_language else None
    )
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

    if task_name in ["S2ST", "S2TT", "ASR"]:
        if audio_source == "microphone":
            input_data = input_audio_mic
        else:
            input_data = input_audio_file

        arr, org_sr = torchaudio.load(input_data)
        new_arr = torchaudio.functional.resample(
            arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE
        )

        max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
        if new_arr.shape[1] > max_length:
            new_arr = new_arr[:, :max_length]
            gr.Warning(
                f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used."
            )
        torchaudio.save(input_data, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
    else:
        input_data = input_text

    text_out, wav, sr = translator.predict(
        input=input_data,
        task_str=task_name,
        tgt_lang=target_language_code,
        src_lang=source_language_code,
        ngram_filtering=True,
    )

    if task_name in ["S2ST", "T2ST"]:
        return (sr, wav.cpu().detach().numpy()), text_out
    else:
        return None, text_out


"""
def predict(
    task_name: str,
    audio_source: str,
    input_audio_mic: str | None,
    input_audio_file: str | None,
    input_text: str | None,
    source_language: str | None,
    target_language: str,
    stream: np.ndarray | None
) -> tuple[np.ndarray, tuple[int, np.ndarray] | None, str]:
    task_name = task_name.split()[0]
    source_language_code = (
        LANGUAGE_NAME_TO_CODE[source_language] if source_language else None
    )
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

    if task_name in ["S2ST", "S2TT", "ASR"]:
        if audio_source == "microphone":
            input_data = input_audio_mic
        else:
            input_data = input_audio_file

        arr, org_sr = torchaudio.load(input_data)
        new_arr = torchaudio.functional.resample(
            arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE
        )

        # Append the arr to the stream to maintain the state
        if stream is None:
            stream = new_arr
        else:
            stream = np.concatenate([stream, new_arr])
        new_arr = stream


        max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
        if new_arr.shape[1] > max_length:
            new_arr = new_arr[:, :max_length]
            gr.Warning(
                f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used."
            )
        torchaudio.save(input_data, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
    else:
        input_data = input_text
    text_out, wav, sr = translator.predict(
        input=input_data,
        task_str=task_name,
        tgt_lang=target_language_code,
        src_lang=source_language_code,
        ngram_filtering=True,
    )
    if task_name in ["S2ST", "T2ST"]:
        return stream, (sr, wav.cpu().detach().numpy()), text_out
    else:
        return stream, None, text_out

"""


def stream_text_check_var_func(
        stream_text_check_var: bool,
        input_text: str
) -> bool:
    if len(input_text) > 3:
        return True


def add_to_text_stream(
        input_text: str,
        instreams: list
) -> tuple[str, str | list[list[str | None]]]:
    if instreams is None:
        final_text = input_text
    else:
        final_text = instreams + [[input_text, None]]
    return "", final_text


def stream_audio(audio_file, lag, text_out):
    audio = AudioSegment.from_wav(audio_file)
    i = 0
    chunk_size = 1000

    while chunk_size * i < len(audio):
        chunk = audio[chunk_size * i:chunk_size * (i + 1)]
        i += 1
        if chunk:
            file = f"/tmp/{i}.mp3"
            chunk.export(file, format="mp3")
            yield file, text_out


""""
def stream_audio(lag):   #for T2ST
    audio_file = 'test.mp3'  # Your audio file path
    audio = AudioSegment.from_mp3(audio_file)
    chunk_length = 1000
    chunks = []
    while len(audio) > chunk_length:
        chunks.append(audio[:chunk_length])
        audio = audio[chunk_length:]
    if len(audio):  # Ensure we don't end up with an empty chunk
        chunks.append(audio)

    def iter_chunks():
        #https://github.com/gradio-app/gradio/pull/5077
        for chunk in chunks:
            file_like_object = chunk.export(format="mp3")
            data = file_like_object.read()
            time.sleep(lag)
            yield data, "fixed response"

    #return iter_chunks(), "fixed response"
"""


def add_to_stream(audio, instream):
    time.sleep(1)
    if audio is None:
        return gr.update(), instream

    print(f"instream is {instream}")
    print(f"audio is {audio}")
    arr_audio, org_sr_audio = torchaudio.load(audio)
    new_arr = torchaudio.functional.resample(
        arr_audio, orig_freq=org_sr_audio, new_freq=AUDIO_SAMPLE_RATE
    )
    new_arr = new_arr.type(torch.float32)
    new_arr /= torch.max(torch.abs(new_arr))

    if instream is None:
        # print(new_arr)
        # print(org_sr_audio)
        # print(org_sr_audio, new_arr)
        ret = (org_sr_audio, new_arr)
    else:
        org_sr_instream, arr_instream = instream
        # print(np.shape(arr_instream))
        # print(np.shape(arr_audio))

        ret = (org_sr_instream, torch.concatenate([arr_instream, arr_audio], axis=1))
    return ret


def streaming_speech_2_text(
        task_name: str,
        control_source: str,
        audio_source: str,
        input_audio_mic: str | None,
        input_audio_file: str | None,
        input_text: str | None,
        source_language: str | None,
        target_language: str,
        streams
) -> str:
    pass


def streaming_text(
        task_name: str,
        control_source: str,
        audio_source: str,
        input_audio_mic: str | None,
        input_audio_file: str | None,
        input_text: str | None,
        source_language: str | None,
        target_language: str,
        streams
) -> str:
    task_name = task_name.split()[0]
    if control_source == "translate":
        yield "Please click Translate"
    response = None
    string_response = None
    if task_name == "T2TT":
        _, response = predict(
            task_name=task_name,
            audio_source="",
            input_audio_mic=None,
            input_audio_file=None,
            input_text=input_text,
            source_language=source_language,
            target_language=target_language,
        )
        byte_response = response.bytes()
        string_response = byte_response.decode("utf-8")
        yield None, string_response

    elif task_name == "S2TT":
        input_data = input_audio_mic
        org_sr, new_arr = streams
        # new_arr_tensor = torch.from_numpy(new_arr)
        torchaudio.save(input_data, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
        print(f"input_data {input_data}")
        _, response = predict(
            task_name=task_name,
            audio_source="",
            input_audio_mic=None,
            input_audio_file=input_data,
            input_text=None,
            source_language=source_language,
            target_language=target_language,
        )
        string_response = response
        yield None, string_response

    elif task_name == "T2ST":
        (sr, np_arr), text_out = predict(
            task_name=task_name,
            audio_source="",
            input_audio_mic=None,
            input_audio_file=None,
            input_text=input_text,
            source_language=source_language,
            target_language=target_language,
        )

        byte_response = text_out.bytes()
        string_response = byte_response.decode("utf-8")
        print(string_response)
        write("audio_file", sr, np_arr, normalized=False)
        audio = AudioSegment.from_mp3("audio_file")
        lag = 2

        chunk_length = 1000
        chunks = []
        while len(audio) > chunk_length:
            chunks.append(audio[:chunk_length])
            audio = audio[chunk_length:]
        if len(audio):  # Ensure we don't end up with an empty chunk
            chunks.append(audio)

        for chunk in chunks:
            file_like_object = chunk.export(format="mp3")
            data = file_like_object.read()
            time.sleep(lag)
            yield data, string_response


    else:
        print(f"In streaming text {task_name}")

    if response is None:
        string_response = "Nothing yet in the stream"
    """
    # This is to split into chunks and then make a generator out of it.
    string_response_split = string_response.split(" ")
    string_response_3_split_words = [string_response_split[i:i+3] for i in range(0, len(string_response_split), 3)]
    string_response_3_joined_words = [" ".join(word) for word in string_response_3_split_words]
    print(f"string_response_3_joined_words is {string_response_3_joined_words}")

    for partial_sentence in string_response_3_joined_words:
        history += partial_sentence + " "
        time.sleep(0.0000001)
        print(f"history is {history}")
        yield history
    """
    yield None, string_response


def write(audio_file, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(audio_file, format="mp3", bitrate="320k")


def process_s2st_example(
        input_audio_file: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="S2ST",
        audio_source="file",
        input_audio_mic=None,
        input_audio_file=input_audio_file,
        input_text=None,
        source_language=None,
        target_language=target_language,
    )


def process_s2tt_example(
        input_audio_file: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="S2TT",
        audio_source="file",
        input_audio_mic=None,
        input_audio_file=input_audio_file,
        input_text=None,
        source_language=None,
        target_language=target_language,
    )


def process_t2st_example(
        input_text: str, source_language: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="T2ST",
        audio_source="",
        input_audio_mic=None,
        input_audio_file=None,
        input_text=input_text,
        source_language=source_language,
        target_language=target_language,
    )


def process_t2tt_example(
        input_text: str, source_language: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="T2TT",
        audio_source="",
        input_audio_mic=None,
        input_audio_file=None,
        input_text=input_text,
        source_language=source_language,
        target_language=target_language,
    )


def process_asr_example(
        input_audio_file: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="ASR",
        audio_source="file",
        input_audio_mic=None,
        input_audio_file=input_audio_file,
        input_text=None,
        source_language=None,
        target_language=target_language,
    )


# Functions to be outputed from the event listeners
def update_control_source_ui(control_source: str) -> gr.Dropdown:
    control_source = control_source == "streaming"
    if control_source:
        return (
            gr.Dropdown(choices=["microphone"])
        )
    else:
        return (
            gr.Dropdown(choices=["microphone", "file"])
        )


def update_audio_ui(
        audio_source: str,
        control_source: str) -> tuple[dict, dict]:
    mic = audio_source == "microphone"
    translate = control_source == "translate"
    print(f"translate is {translate}")
    return (
        gr.update(visible=mic, value=None) if translate else gr.update(visible=mic, streaming=True),  # input_audio_mic
        gr.update(visible=not mic, value=None),  # input_audio_file
    )


def update_input_ui(task_name: str,
                    control_source: str) -> tuple[dict, dict, dict, dict, dict]:
    task_name = task_name.split()[0]
    print(f"task name is {task_name} and control_source is {control_source}")
    if task_name == "S2ST":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True,
                choices=S2ST_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
            gr.update(visible=True) if control_source == 'translate' else gr.update(visible=False)  # btn
        )
    elif task_name == "S2TT":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True,
                choices=S2TT_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
            gr.update(visible=True) if control_source == 'translate' else gr.update(visible=False)  # btn

        )
    elif task_name == "T2ST":
        return (
            gr.update(visible=False),  # audio_box
            gr.update(visible=True,
                      placeholder="Click Translate to submit") if control_source == "translate" else gr.update(
                visible=True, placeholder="Continuously type"),  # input_text
            gr.update(visible=True),  # source_language
            gr.update(
                visible=True,
                choices=S2ST_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
            gr.update(visible=True) if control_source == 'translate' else gr.update(visible=False)  # btn

        )
    elif task_name == "T2TT":
        return (
            gr.update(visible=False),  # audio_box
            gr.update(visible=True,
                      placeholder="Click Translate to submit") if control_source == "translate" else gr.update(
                visible=True, placeholder="Continuously type"),  # input_text
            gr.update(visible=True),  # source_language
            gr.update(
                visible=True,
                choices=T2TT_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
            gr.update(visible=True) if control_source == 'translate' else gr.update(visible=False)  # btn

        )
    elif task_name == "ASR":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True,
                choices=S2TT_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
            gr.update(visible=True) if control_source == 'translate' else gr.update(visible=False)  # btn

        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


def update_output_ui(
        task_name: str,
        control_source: str) -> tuple[dict, dict]:
    task_name = task_name.split()[0]
    translate = control_source == 'translate'
    if task_name in ["S2ST", "T2ST"]:
        return (
            gr.update(visible=True, value=None) if translate else gr.update(visible=True, streaming=True,
                                                                            autoplay=True),  # output_audio
            gr.update(value=None),  # output_text
        )
    elif task_name in ["S2TT", "T2TT", "ASR"]:
        return (
            gr.update(visible=False, value=None),  # output_audio
            gr.update(value=None),  # output_text
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


def update_example_ui(task_name: str) -> tuple[dict, dict, dict, dict, dict]:
    task_name = task_name.split()[0]
    return (
        gr.update(visible=task_name == "S2ST"),  # s2st_example_row
        gr.update(visible=task_name == "S2TT"),  # s2tt_example_row
        gr.update(visible=task_name == "T2ST"),  # t2st_example_row
        gr.update(visible=task_name == "T2TT"),  # t2tt_example_row
        gr.update(visible=task_name == "ASR"),  # asr_example_row
    )


css = """
h1 {
  text-align: center;
}

.contain {
  max-width: 730px;
  margin: auto;
  padding-top: 1.5rem;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    streams = gr.State()
    with gr.Group():
        task_name = gr.Dropdown(
            label="Task",
            choices=TASK_NAMES,
            value=TASK_NAMES[0],
        )
        with gr.Row():
            source_language = gr.Dropdown(
                label="Source language",
                choices=TEXT_SOURCE_LANGUAGE_NAMES,
                value="English",
                visible=False,
            )
            target_language = gr.Dropdown(
                label="Target language",
                choices=S2ST_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            )
        with gr.Row() as control_box:
            control_source = gr.Radio(
                label="Control source",
                choices=["streaming", "translate"],
                value="streaming",
            )

        with gr.Row() as audio_box:
            audio_source = gr.Dropdown(
                label="Audio source",
                choices=["microphone"],
                value="microphone",
            )
            input_audio_mic = gr.Audio(
                label="Input speech",
                type="filepath",
                source="microphone",
                visible=True,
            )
            input_audio_file = gr.Audio(
                label="Input speech",
                type="filepath",
                source="upload",
                visible=False,
            )
        input_text = gr.Textbox(label="Input text", visible=False)
        btn = gr.Button("Translate", visible=False)
        with gr.Column():
            output_audio = gr.Audio(
                label="Translated speech",
                autoplay=False,
                streaming=False,
                type="numpy",
            )
            output_text = gr.Textbox(label="Translated text")

        with gr.Row(visible=True) as s2st_example_row:
            s2st_examples = gr.Examples(
                examples=[
                    ["assets/sample_input.mp3", "French"],
                    ["assets/sample_input.mp3", "Mandarin Chinese"],
                    ["assets/sample_input_2.mp3", "Hindi"],
                    ["assets/sample_input_2.mp3", "Spanish"],
                ],
                inputs=[input_audio_file, target_language],
                outputs=[output_audio, output_text],
                fn=process_s2st_example,
            )
        with gr.Row(visible=False) as s2tt_example_row:
            s2tt_examples = gr.Examples(
                examples=[
                    ["assets/sample_input.mp3", "French"],
                    ["assets/sample_input.mp3", "Mandarin Chinese"],
                    ["assets/sample_input_2.mp3", "Hindi"],
                    ["assets/sample_input_2.mp3", "Spanish"],
                ],
                inputs=[input_audio_file, target_language],
                outputs=[output_audio, output_text],
                fn=process_s2tt_example,
            )
        with gr.Row(visible=False) as t2st_example_row:
            t2st_examples = gr.Examples(
                examples=[
                    ["My favorite animal is the elephant.", "English", "French"],
                    ["My favorite animal is the elephant.", "English", "Mandarin Chinese"],
                    [
                        "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                        "English",
                        "Hindi",
                    ],
                    [
                        "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                        "English",
                        "Spanish",
                    ],
                ],
                inputs=[input_text, source_language, target_language],
                outputs=[output_audio, output_text],
                fn=process_t2st_example,
            )
        with gr.Row(visible=False) as t2tt_example_row:
            t2tt_examples = gr.Examples(
                examples=[
                    ["My favorite animal is the elephant.", "English", "French"],
                    ["My favorite animal is the elephant.", "English", "Mandarin Chinese"],
                    [
                        "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                        "English",
                        "Hindi",
                    ],
                    [
                        "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                        "English",
                        "Spanish",
                    ],
                ],
                inputs=[input_text, source_language, target_language],
                outputs=[output_audio, output_text],
                fn=process_t2tt_example,
            )
        with gr.Row(visible=False) as asr_example_row:
            asr_examples = gr.Examples(
                examples=[
                    ["assets/sample_input.mp3", "English"],
                    ["assets/sample_input_2.mp3", "English"],
                ],
                inputs=[input_audio_file, target_language],
                outputs=[output_audio, output_text],
                fn=process_asr_example,
            )

        # Event listeners
        control_source.change(
            fn=update_control_source_ui,
            inputs=control_source,
            outputs=audio_source,
            queue=False,
            api_name=False
        )
        audio_source.change(
            fn=update_audio_ui,
            inputs=[audio_source, control_source],
            outputs=[
                input_audio_mic,
                input_audio_file,
            ],
            queue=False,
            api_name=False,
        )

        # change input ui and output ui based on both control_source and task_name
        task_name.change(
            fn=update_input_ui,
            inputs=[task_name, control_source],
            outputs=[
                audio_box,
                input_text,
                source_language,
                target_language,
                btn
            ],
            queue=False,
            api_name=False,
        ).then(
            fn=update_output_ui,
            inputs=task_name,
            outputs=[output_audio, output_text],
            queue=False,
            api_name=False,
        ).then(
            fn=update_example_ui,
            inputs=task_name,
            outputs=[
                s2st_example_row,
                s2tt_example_row,
                t2st_example_row,
                t2tt_example_row,
                asr_example_row,
            ],
            queue=False,
            api_name=False,
        )

        control_source.change(
            fn=update_input_ui,
            inputs=[task_name, control_source],
            outputs=[
                audio_box,
                input_text,
                source_language,
                target_language,
                btn
            ],
            queue=False,
            api_name=False,
        ).then(
            fn=update_output_ui,
            inputs=task_name,
            outputs=[output_audio, output_text],
            queue=False,
            api_name=False,
        ).then(
            fn=update_example_ui,
            inputs=task_name,
            outputs=[
                s2st_example_row,
                s2tt_example_row,
                t2st_example_row,
                t2tt_example_row,
                asr_example_row,
            ],
            queue=False,
            api_name=False,
        )

        """
        input_audio_file.upload(
            fn=predict,
            inputs=[
                task_name,
                audio_source,
                input_audio_mic,
                input_audio_file,
                input_text,
                source_language,
                target_language,
            ],
            outputs=[output_audio, output_text],
            api_name=False,
        )

        input_audio_mic.stop_recording(
            fn=predict,
            inputs=[
                task_name,
                audio_source,
                input_audio_mic,
                input_audio_file,
                input_text,
                source_language,
                target_language,
            ],
            outputs=[output_audio, output_text],
            api_name=False,
        )

        input_text.submit(
            fn=predict,
            inputs=[
                task_name,
                audio_source,
                input_audio_mic,
                input_audio_file,
                input_text,
                source_language,
                target_language,
            ],
            outputs=[output_audio, output_text],
            api_name=False,
        )
        """
        btn.click(
            fn=predict,
            inputs=[
                task_name,
                audio_source,
                input_audio_mic,
                input_audio_file,
                input_text,
                source_language,
                target_language,
            ],
            outputs=[output_audio, output_text],
            api_name="run",
        )

        # Todo
        # streaming for S2TT

        input_audio_mic.stream(
            fn=add_to_stream,
            inputs=[input_audio_mic, streams],
            outputs=[streams],
            queue=False
        ).then(
            fn=streaming_text,
            inputs=[task_name,
                    control_source,
                    audio_source,
                    input_audio_mic,
                    input_audio_file,
                    input_text,
                    source_language,
                    target_language,
                    streams
                    ],
            outputs=[output_text],
            queue=False
        )

        # streaming fot T2TT
        input_text.change(
            fn=streaming_text,
            inputs=[task_name,
                    control_source,
                    audio_source,
                    input_audio_mic,
                    input_audio_file,
                    input_text,
                    source_language,
                    target_language,
                    streams
                    ],
            outputs=[output_audio, output_text],
            queue=False
        )

    if __name__ == '__main__':
        demo.queue().launch(share=True, debug=True)