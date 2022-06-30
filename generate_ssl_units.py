import pytorch_lightning as pl
from tqdm import tqdm
import s3prl.hub as hub
import pickle
import random

from lightning.utils.tool import torch_exist_nan, read_queries_from_txt
from lightning.systems import get_system

from dlhlp_lib.parsers.preprocess import *
from dlhlp_lib.audio import AUDIO_CONFIG
import Define
from Parsers.parser import DataParser
from dpdp import DPDP


INV_FRAME_PERIOD = AUDIO_CONFIG["audio"]["sampling_rate"] / AUDIO_CONFIG["stft"]["hop_length"]


def generate_ssl_centroids(n_clusters, unit_name: str, root: str, dpdp: DPDP, src_txt_path: str):
    data_parser = DataParser(root)
    queries = read_queries_from_txt(src_txt_path)
    queries = random.sample(queries, 512)
    wav_paths = [data_parser.wav_trim_16000.read_filename(q, raw=True) for q in queries]

    data_parser.create_ssl_unit_feature(unit_name=unit_name)

    # KMeans
    kmeans_model = dpdp.calculate_ssl_centroids(n_clusters, wav_paths)
    output_path = f"{data_parser.ssl_units[unit_name].root}/centroids.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(kmeans_model, f)


def generate_ssl_units(unit_name: str, root: str, dpdp: DPDP):
    data_parser = DataParser(root)
    queries = data_parser.get_all_queries()

    # DPDP
    unit_parser = data_parser.ssl_units[unit_name]
    segment_feat = unit_parser.dp_segment
    phoneme_feat = unit_parser.phoneme

    centroids_path = f"{data_parser.ssl_units[unit_name].root}/centroids.pkl"
    with open(centroids_path, 'rb') as f:
        kmeans_model = pickle.load(f)
    
    for query in tqdm(queries):
        try:
            wav_path = data_parser.wav_trim_16000.read_filename(query, raw=True)
            segment, phoneme = dpdp.segment(wav_path, kmeans_model)
            segment_feat.save(segment, query)
            phoneme = [str(phn) for phn in phoneme]
            phoneme_feat.save(" ".join(phoneme), query)
        except:
            print(query)

    # Other preprocessing
    segment2duration_mp(unit_parser, queries, "dp_segment", "dp_duration", INV_FRAME_PERIOD, n_workers=os.cpu_count() // 2, refresh=True)
    duration_avg_pitch_and_energy_mp(data_parser, queries, f"ssl_units/{unit_name}/dp_duration", n_workers=os.cpu_count() // 2, refresh=True)


class CodebookPostnet(pl.LightningModule):
    def __init__(self, system_type: str, ckpt_path: str) -> None:
        super().__init__()
        # TODO: Model initialization dependent on global initialization, should improve design
        keys = []
        with open("preprocessed_data/JSUT/stats.json") as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"]
            Define.ALLSTATS["JSUT"] = stats
            keys.append("JSUT")
        Define.ALLSTATS["global"] = Define.merge_stats(Define.ALLSTATS, keys)

        system = get_system(system_type)
        self.fscl_model = system.load_from_checkpoint(ckpt_path).cuda()
        
    def forward(self, repr):
        repr = repr.transpose(1, 2).cuda()
        return self.fscl_model.embedding_model.get_new_embedding(self.fscl_model.codebook_type, 
                        ref_phn_feats=repr, lang_id=6).detach().cpu()


if __name__ == "__main__":
    extractor = getattr(hub, 'hubert_large_ll60k')().cuda()
    extractor.eval()
    codebook_postnet = CodebookPostnet(
        system_type="semi-fscl",
        ckpt_path="output/ckpt/fscl/fa354467dffc4d8b843764069974a191/checkpoints/epoch=19-step=50000.ckpt"
    )
    dpdp = DPDP(extractor, layer=24, fp=20, norm=False, postnet=codebook_postnet)

    # generate_ssl_centroids(32, "hubert-32", "./preprocessed_data/JSUT", dpdp, src_txt_path="_data/JSUT/train.txt")
    # generate_ssl_units("hubert-32", "./preprocessed_data/JSUT", dpdp)

    # generate_ssl_centroids(48, "hubert-48", "./preprocessed_data/JSUT", dpdp, src_txt_path="_data/JSUT/train.txt")
    # generate_ssl_units("hubert-48", "./preprocessed_data/JSUT", dpdp)

    # generate_ssl_centroids(64, "hubert-64", "./preprocessed_data/JSUT", dpdp, src_txt_path="_data/JSUT/train.txt")
    # generate_ssl_units("hubert-64", "./preprocessed_data/JSUT", dpdp)

    # generate_ssl_centroids(96, "hubert-96", "./preprocessed_data/JSUT", dpdp, src_txt_path="_data/JSUT/train.txt")
    # generate_ssl_units("hubert-96", "./preprocessed_data/JSUT", dpdp)

    # generate_ssl_centroids(128, "hubert-128", "./preprocessed_data/JSUT", dpdp, src_txt_path="_data/JSUT/train.txt")
    # generate_ssl_units("hubert-128", "./preprocessed_data/JSUT", dpdp)

    generate_ssl_centroids(32, "hubert-codebook-32", "./preprocessed_data/JSUT", dpdp, src_txt_path="_data/JSUT/train.txt")
    generate_ssl_units("hubert-codebook-32", "./preprocessed_data/JSUT", dpdp)
