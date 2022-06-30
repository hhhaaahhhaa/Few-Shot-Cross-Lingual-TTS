import numpy as np
import torch
from tqdm import tqdm
import s3prl.hub as hub
import pickle
import librosa
import random
from typing import List
import time

from lightning.utils.tool import torch_exist_nan, read_queries_from_txt
from sklearn.cluster import KMeans


class DPDP(object):

    SAMPLE_RATE = 16000

    @staticmethod
    def default_pen(segment_length):
        return 1 - segment_length

    def __init__(self, extractor, layer, fp=20, norm=False, postnet=None):
        self._extractor = extractor
        self._layer = layer
        self._fp = fp
        self._norm = norm
        self._pen = DPDP.default_pen

        self.postnet = self.__slice_func
        if postnet is not None:
            self.postnet = postnet
    
    def set_penalize_function(self, func):
        self._pen = func

    def __slice_func(self, repr):
        return repr[:, self._layer]
    
    def calculate_ssl_centroids(self, n_clusters: int, wav_paths: List[str], batch_size=16) -> KMeans:
        all_frames = []
        data = (wav_paths,)
        gen = batch_ssl_extraction_generator(data, self._extractor, fp=self._fp, batch_size=batch_size, norm=self._norm)
        
        for (batch_paths, representation, n_frames) in tqdm(gen):
            representation = self.postnet(representation)
            for wav_path, repr, n_frame in zip(batch_paths, representation, n_frames):
                sliced_repr = repr[:n_frame, :].clone()  # L, dim
                # print(sliced_repr.shape)
                try:
                    assert not torch_exist_nan(sliced_repr)
                except:
                    self.log("NaN in SSL feature:")
                    self.log(wav_path)
                    continue
                all_frames.append(sliced_repr.numpy())

        # Concatenate and perform KMeans clustering.
        st = time.time()
        self.log("Perform KMeans...")
        all_frames = np.concatenate(all_frames, axis=0)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_frames)
        self.log(f"Done in {time.time()-st:.2f}s. Data {all_frames.shape} => Centroids {kmeans.cluster_centers_.shape}.")

        return kmeans

    def batch_segment(self, wav_paths: List[str], kmeans_model: KMeans, batch_size=16, lambd=35):
        """
        Performing segment with a batch of paths is not done yet!
        """
        data = (wav_paths,)
        gen = batch_ssl_extraction_generator(data, self._extractor, fp=self._fp, batch_size=batch_size, norm=self._norm)
        
        for (batch_paths, representation, n_frames) in tqdm(gen):
            for wav_path, repr, n_frame in zip(batch_paths, representation, n_frames):
                sliced_repr = repr[self._layer, :n_frame, :].clone()  # L, dim
                try:
                    assert not torch_exist_nan(sliced_repr)
                except:
                    self.log("NaN in SSL feature:")
                    self.log(wav_path)
                    continue
                sliced_repr = sliced_repr.numpy()
                tokens = kmeans_model.predict(sliced_repr).tolist()
                self.log(f"tokens: {tokens}")
                self.log(f"len(tokens): {len(tokens)}")

                boundaries, label_tokens = segment(sliced_repr, kmeans_model, self._pen, lambd=lambd)
                self.log(f"boundaries: {boundaries}")
                self.log(f"label_tokens: {label_tokens}")
                self.log(f"Num of segments = {len(label_tokens)}")

    def segment(self, wav_path: str, kmeans_model: KMeans, lambd=35):
        # Support .wav or .npy input format
        if wav_path[-4:] == ".wav":
            wav, sr = librosa.load(wav_path, sr=None)
            assert sr == 16000, "Sample rate need to be 16kHz."
        elif wav_path[-4:] == ".npy":
            wav = np.load(wav_path)
        else:
            raise NotImplementedError
        wavs = [torch.from_numpy(wav).float().cuda()]
        
        with torch.no_grad():
            representation = self._extractor(wavs)
            representation = torch.stack(representation["hidden_states"], dim=1)  # 1, layer, L, dim
            if self._norm:
                representation = torch.nn.functional.normalize(representation, dim=3)
            representation = representation.detach().cpu()
        
        representation = self.postnet(representation)
        sliced_repr = representation[0].clone()  # L, dim
        try:
            assert not torch_exist_nan(sliced_repr)
        except:
            self.log("NaN in SSL feature:")
            self.log(wav_path)
            raise ValueError
        
        sliced_repr = sliced_repr.numpy()
        # tokens = kmeans_model.predict(sliced_repr).tolist()
        # self.log(f"tokens: {tokens}")
        # self.log(f"len(tokens): {len(tokens)}")

        boundaries, label_tokens = segment(sliced_repr, kmeans_model, self._pen, lambd=lambd)
        # self.log(f"boundaries: {boundaries}")
        # self.log(f"label_tokens: {label_tokens}")
        # self.log(f"Num of segments = {len(label_tokens)}")

        foramtted_boundaries = []
        st = 0.0
        for b in boundaries:
            foramtted_boundaries.append((st, b * self._fp / 1000))
            st = b * self._fp / 1000
        
        return foramtted_boundaries, label_tokens
    
    def log(self, msg):
        print(f"[DPDP]: ", msg)


def batch_ssl_extraction_generator(data, extractor, fp=20, batch_size=16, norm=False, shuffle=False):
    """
    Batch generator for ssl extraction.
    """
    wav_paths, *other = data
    n_samples = len(wav_paths)
    indices = np.arange(n_samples)
    if shuffle:  # Shuffle at the start of epoch
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]

        wavs = []
        n_frames = []
        batch_paths = [wav_paths[idx] for idx in batch_idx]
        for wav_path in batch_paths:
            # Support .wav or .npy input format
            if wav_path[-4:] == ".wav":
                wav, sr = librosa.load(wav_path, sr=None)
                assert sr == 16000, "Sample rate need to be 16kHz."
            elif wav_path[-4:] == ".npy":
                wav = np.load(wav_path)
            else:
                raise NotImplementedError

            wavs.append(torch.from_numpy(wav).float().cuda())
            n_frames.append(len(wav) // (16 * fp))
        
        with torch.no_grad():
            representation = extractor(wavs)
            representation = torch.stack(representation["hidden_states"], dim=1)  # bs, layer, L, dim
            if norm:
                representation = torch.nn.functional.normalize(representation, dim=3)
            representation = representation.detach().cpu()
        
        remain = ([feat[idx] for idx in batch_idx] for feat in other)

        yield (batch_paths, representation, n_frames, *remain)


# Simple implementation of dynamic programming based phoneme segmentation method given in
#   Towards unsupervised phone and word segmentation using self-supervised vector-quantized neural networks
#   (https://arxiv.org/abs/2012.07551, INTERSPEECH 2021)
# Author: Yuan Tseng (https://github.com/roger-tseng)
def segment(reps, kmeans_model, pen, lambd=35):
    '''
    Inputs:
    reps        :   Representation sequence from self supervised model
    kmeans_model:   Pretrained scikit-learn MiniBatchKMeans model
    pen         :   penalty function penalizing segment length (longer segment, higher penalty)
    lambd       :   penalty weight (larger weight, longer segment)

    Outputs:
    boundaries  :   List of tokens at right boundaries of segments 
                    (assuming token sequence starts from 1 to Tth token)
    label_token :   List of token labels for segments

    e.g. :

    If  tokens = [34, 55, 62, 83, 42]
        boundaries = [3, 5]
        label_token = [55, 83]

    then segmentation is :
    | 34 55 62 | 83 42 |
    |    55    |   83  |

    '''
    
    # array of distances to closest cluster center, size: token sequence len * num of clusters
    distance_array = np.square( kmeans_model.transform(reps) )
    alphas = [[0, None]]

    # Perform dynamic-programming-based segmentation
    for t in range(1,reps.shape[0]+1):

        errors = []
        closest_centers = []
        
        for segment_length in range(1,t+1):

            # array len = num of clusters
            # ith element is sum of distance from the last segment_length tokens until Tth token to the ith cluster center
            distance_subarray = distance_array[t-segment_length:t].sum(axis=0)

            closest_center = distance_subarray.argmin()
            error = alphas[t-segment_length][0] + distance_subarray.min() + lambd * pen(segment_length)

            closest_centers.append(closest_center)
            errors.append(error)

        errors = np.array(errors)
        alpha, a_min, closest = errors.min(), t-1-errors.argmin(), closest_centers[errors.argmin()]
        alphas.append([alpha, a_min, closest])

    # Backtrack to find optimal boundary tokens and label
    boundaries = []
    label_tokens = []
    tk = len(alphas)-1
    while (tk!=0):
        boundaries.append(tk)
        label_tokens.append(alphas[tk][2])
        tk = alphas[tk][1]  
    boundaries.reverse()
    label_tokens.reverse()

    return boundaries, label_tokens


if __name__ == "__main__":
    from Parsers.parser import DataParser
    data_parser = DataParser("/mnt/d/Projects/Few-Shot-Cross-Lingual-TTS/preprocessed_data/JSUT")
    input_path = "_data/JSUT/train.txt"
    output_path = "_data/JSUT/hubert-kmeans.pkl"

    extractor = getattr(hub, 'hubert_large_ll60k')().cuda()
    extractor.eval()
    dpdp = DPDP(extractor, layer=24, fp=20)
    queries = read_queries_from_txt(input_path)
    queries = random.sample(queries, 512)
    wav_paths = [data_parser.wav_trim_16000.read_filename(q, raw=True) for q in queries]

    # KMeans
    kmeans_model = dpdp.calculate_ssl_centroids(64, wav_paths, batch_size=16)
    with open(output_path, 'wb') as f:
        pickle.dump(kmeans_model, f)

    # DPDP segmentation
    with open(output_path, 'rb') as f:
        kmeans_model = pickle.load(f)
    dpdp.segment(wav_paths, kmeans_model)
