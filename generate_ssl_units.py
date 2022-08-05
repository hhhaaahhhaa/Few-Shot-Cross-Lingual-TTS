import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import s3prl.hub as hub
import pickle
import random
from sklearn.cluster import KMeans

from lightning.utils.tool import read_queries_from_txt
from lightning.systems import get_system

from dlhlp_lib.algorithm.dpdp import DPDPSSLUnit
from dlhlp_lib.parsers.preprocess import *
from dlhlp_lib.audio import AUDIO_CONFIG
import Define
from Parsers.parser import DataParser


INV_FRAME_PERIOD = AUDIO_CONFIG["audio"]["sampling_rate"] / AUDIO_CONFIG["stft"]["hop_length"]


def generate_ssl_centroids(n_clusters, unit_name: str, root: str, dpdp: DPDPSSLUnit, src_txt_path: str):
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


def generate_ssl_centroids_from_array(unit_name: str, root: str, array: np.array, postnet=None):
    data_parser = DataParser(root)
    data_parser.create_ssl_unit_feature(unit_name=unit_name)

    # KMeans
    if postnet is not None:
        array = postnet(torch.from_numpy(np.expand_dims(array, axis=0))).detach().cpu().numpy()
        array = array[0]
    n_clusters, n_features = array.shape
    kmeans_model = KMeans(n_clusters=n_clusters, init=array, max_iter=1)  # just run one k-Means iteration so that the centroids are not updated
    kmeans_model.fit(np.zeros((n_clusters, n_features)))
    kmeans_model.cluster_centers_ = array
    output_path = f"{data_parser.ssl_units[unit_name].root}/centroids.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(kmeans_model, f)


def generate_ssl_units(unit_name: str, root: str, dpdp: DPDPSSLUnit):
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
            segment, phoneme = dpdp.segment(wav_path, kmeans_model, lambd=10)
            segment_feat.save(segment, query)
            phoneme = [str(phn) for phn in phoneme]
            phoneme_feat.save(" ".join(phoneme), query)
        except:
            raise
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
        self.fscl_model = system.load_from_checkpoint(ckpt_path)
        
    def forward(self, repr):
        """
        Pass repr into codebook, input and output have the same size.
        Args:
            repr: Tensor with shape [bs, L, layer, dim].
        """
        repr = repr.to(self.device)
        return self.fscl_model.embedding_model.get_new_embedding(self.fscl_model.codebook_type, 
                        ref_phn_feats=repr, lang_id=6)


if __name__ == "__main__":
    extractor = getattr(hub, 'hubert_large_ll60k')().cuda()
    extractor.eval()
    # codebook_postnet = CodebookPostnet(
    #     system_type="semi-fscl",
    #     ckpt_path="output/ckpt/fscl/fa354467dffc4d8b843764069974a191/checkpoints/epoch=19-step=50000.ckpt"
    # ).cuda()
    # dpdp = DPDP(extractor, layer=24, fp=20, norm=False, postnet=None)
    dpdp = DPDPSSLUnit('hubert_large_ll60k', layer=24, postnet=None)
    dpdp.cuda()

    # ========= debug section =============
    generate_ssl_centroids(32, "debug", "./preprocessed_data/JSUT", dpdp, src_txt_path="_data/JSUT/train.txt")
    generate_ssl_units("debug", "./preprocessed_data/JSUT", dpdp)

    # with open("_data/JSUT/hubert-phoneme-average.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # reprs = [table[p] for p in table]
    # array = np.stack(reprs, axis=0)[:, 24, :]  # n_phns, 25, 1024
    # generate_ssl_centroids_from_array("debug", "./preprocessed_data/JSUT", array=array)
    # # generate_ssl_units("gtcent-hubert-reg10", "./preprocessed_data/JSUT", dpdp)
    # pairs = [(str(i), p[1:]) for i, p in enumerate(table)]
    # with open("./preprocessed_data/JSUT/ssl_units/debug/centroids2phoneme.pkl", 'wb') as f:
    #     pickle.dump(pairs, f)
    #     print(pairs)
    # ========= debug section =============

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

    # generate_ssl_centroids(32, "hubert-codebook-32", "./preprocessed_data/JSUT", dpdp, src_txt_path="_data/JSUT/train.txt")
    # generate_ssl_units("hubert-codebook-32", "./preprocessed_data/JSUT", dpdp)
    
    # with open("_data/JSUT/hubert-phoneme-average.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # reprs = [table[p] for p in table]
    # array = np.stack(reprs, axis=0)
    # generate_ssl_centroids_from_array("gtcent-hubert-codebook-reg10", "./preprocessed_data/JSUT", array=array, postnet=codebook_postnet)
    # generate_ssl_units("gtcent-hubert-codebook-reg10", "./preprocessed_data/JSUT", dpdp)
    # pairs = [(str(i), p[1:]) for i, p in enumerate(table)]
    # with open("./preprocessed_data/JSUT/ssl_units/gtcent-hubert-codebook-reg10/centroids2phoneme.pkl", 'wb') as f:
    #     pickle.dump(pairs, f)
    #     print(pairs)

    # with open("_data/JSUT/hubert-phoneme-4shot.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # reprs = [table[p] for p in table]
    # array = np.stack(reprs, axis=0)[:, 24, :]  # n_phns, 25, 1024
    # generate_ssl_centroids_from_array("gtcent-4shot-hubert-reg10", "./preprocessed_data/JSUT", array=array)
    # generate_ssl_units("gtcent-4shot-hubert-reg10", "./preprocessed_data/JSUT", dpdp)
    # pairs = [(str(i), p[1:]) for i, p in enumerate(table)]
    # with open("./preprocessed_data/JSUT/ssl_units/gtcent-4shot-hubert-reg10/centroids2phoneme.pkl", 'wb') as f:
    #     pickle.dump(pairs, f)
    #     print(pairs)

    # with open("_data/JSUT/hubert-phoneme-average.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # reprs = [table[p] for p in table]
    # array = np.stack(reprs, axis=0)[:, 24, :]  # n_phns, 25, 1024
    # generate_ssl_centroids_from_array("gtcent-hubert-reg10", "./preprocessed_data/JSUT", array=array)
    # # generate_ssl_units("gtcent-hubert-reg10", "./preprocessed_data/JSUT", dpdp)
    # pairs = [(str(i), p[1:]) for i, p in enumerate(table)]
    # with open("./preprocessed_data/JSUT/ssl_units/gtcent-hubert-reg10/centroids2phoneme.pkl", 'wb') as f:
    #     pickle.dump(pairs, f)
    #     print(pairs)

    # with open("_data/JSUT/hubert-phoneme-average.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # reprs = [table[p] for p in table]
    # array = np.stack(reprs, axis=0)[:, 24, :]  # n_phns, 25, 1024
    # generate_ssl_centroids_from_array("gtcent-hubert-reg20", "./preprocessed_data/JSUT", array=array)
    # # generate_ssl_units("gtcent-hubert-reg20", "./preprocessed_data/JSUT", dpdp)
    # pairs = [(str(i), p[1:]) for i, p in enumerate(table)]
    # with open("./preprocessed_data/JSUT/ssl_units/gtcent-hubert-reg20/centroids2phoneme.pkl", 'wb') as f:
    #     pickle.dump(pairs, f)
    #     print(pairs)

    # with open("_data/JSUT/hubert-phoneme-average.pkl", 'rb') as f:
    #     table = pickle.load(f)
    # reprs = [table[p] for p in table]
    # array = np.stack(reprs, axis=0)[:, 24, :]  # n_phns, 25, 1024
    # generate_ssl_centroids_from_array("gtcent-hubert", "./preprocessed_data/JSUT", array=array)
    # # generate_ssl_units("gtcent-hubert", "./preprocessed_data/JSUT", dpdp)
    # pairs = [(str(i), p[1:]) for i, p in enumerate(table)]
    # with open("./preprocessed_data/JSUT/ssl_units/gtcent-hubert/centroids2phoneme.pkl", 'wb') as f:
    #     pickle.dump(pairs, f)
    #     print(pairs)
