from codecs import ignore_errors
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import librosa

from dlhlp_lib.audio.tools import wav_normalization
from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.tts_preprocess.basic import *

import Define
from Parsers.parser import DataParser


INV_FRAME_PERIOD = AUDIO_CONFIG["audio"]["sampling_rate"] / AUDIO_CONFIG["stft"]["hop_length"]


def prepare_initial_features(data_parser: DataParser, query, data):
    wav_16000, _ = librosa.load(data["wav_path"], sr=16000)
    wav_22050, _ = librosa.load(data["wav_path"], sr=22050)
    wav_16000 = wav_normalization(wav_16000)
    wav_22050 = wav_normalization(wav_22050)
    data_parser.wav_16000.save(wav_16000, query)
    data_parser.wav_22050.save(wav_22050, query)
    data_parser.text.save(data["text"], query)


def preprocess(data_parser: DataParser, queries):
    ignore_errors = True
    if Define.DEBUG:
        ignore_errors = False
    textgrid2segment_and_phoneme_mp(
        data_parser, queries, 
        textgrid_featname="textgrid",
        segment_featname="mfa_segment",
        phoneme_featname="phoneme",
        ignore_errors=ignore_errors
    )
    trim_wav_by_segment_mp(
        data_parser, queries, sr=22050,
        wav_featname="wav_22050",
        segment_featname="mfa_segment",
        wav_trim_featname="wav_trim_22050",
        refresh=True,
        ignore_errors=ignore_errors
    )
    trim_wav_by_segment_mp(
        data_parser, queries, sr=16000,
        wav_featname="wav_16000",
        segment_featname="mfa_segment",
        wav_trim_featname="wav_trim_16000",
        refresh=True,
        ignore_errors=ignore_errors
    )
    wav_to_mel_energy_pitch_mp(
        data_parser, queries,
        wav_featname="wav_trim_22050",
        mel_featname="mel",
        energy_featname="energy",
        pitch_featname="pitch",
        interp_pitch_featname="interpolate_pitch",
        ignore_errors=ignore_errors
    )
    segment2duration_mp(
        data_parser, queries, inv_frame_period=INV_FRAME_PERIOD,
        segment_featname="mfa_segment",
        duration_featname="mfa_duration",
        refresh=True,
        ignore_errors=ignore_errors
    )
    duration_avg_pitch_and_energy_mp(
        data_parser, queries,
        duration_featname="mfa_duration",
        pitch_featname="pitch",
        energy_featname="energy",
        refresh=True,
        ignore_errors=ignore_errors
    )
    extract_spk_ref_mel_slices_from_wav_mp(
        data_parser, queries, sr=16000,
        wav_featname="wav_trim_16000",
        ref_featname="spk_ref_mel_slices",
        ignore_errors=ignore_errors
    )
