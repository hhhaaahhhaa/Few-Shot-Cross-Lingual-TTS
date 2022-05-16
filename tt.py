import s3prl.hub as hub


ssl_extractor = getattr(hub, "wav2vec2_xlsr")()
ssl_extractor = getattr(hub, "distilhubert")()
ssl_extractor = getattr(hub, "vq_wav2vec")()
ssl_extractor = getattr(hub, "vq_apc")()
ssl_extractor = getattr(hub, "tera")()
ssl_extractor = getattr(hub, "wav2vec2_large_ll60k")()
