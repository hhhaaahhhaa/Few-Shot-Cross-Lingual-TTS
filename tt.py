import torch
from tqdm import tqdm
import s3prl.hub as hub
import pickle

from lightning.utils.tool import torch_exist_nan
from Parsers.parser import DataParser


data_parser = DataParser("./preprocessed_data/LibriTTS")


def go(extractor, ppp):
    batch_size = 1
    queries = data_parser.get_all_queries()[:1024]
    wavs = []
    q_temp = []
    n_frames = 0
    sum_norms = torch.zeros(13)
    for i, query in tqdm(enumerate(queries)):
        try:
            wav = data_parser.wav_trim_16000.read_from_query(query)
        except:
            continue
        wavs.append(torch.from_numpy(wav).float().cuda())
        q_temp.append(query)
        if (i + 1) % batch_size == 0 or i + 1 == len(queries):
            with torch.no_grad():
                representation = extractor(wavs)
                representation = torch.stack(representation["hidden_states"], dim=1)  # bs, layer, L, dim
                for r, q in zip(representation, q_temp):
                    r = r.detach().cpu()
                    try:
                        assert not torch_exist_nan(r)
                    except:
                        print("NaN in SSL feature:")
                        print(q)
                        continue
                    normed = torch.linalg.norm(r, dim=2)
                    n_frames += normed.shape[1]
                    sum_norms += normed.sum(dim=1)
            wavs = []
            q_temp = []

    sum_norms = sum_norms / n_frames
    print(sum_norms)

    with open(ppp, 'wb') as f:
        pickle.dump(sum_norms, f)



if __name__ == "__main__":
    extractor = getattr(hub, 'hubert')().cuda()
    extractor.eval()
    go(extractor, "hubert_layer_norm.pkl")
    extractor = getattr(hub, 'wav2vec2')().cuda()
    extractor.eval()
    go(extractor, "wav2vec2_layer_norm.pkl")
