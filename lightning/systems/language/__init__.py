from .FastSpeech2 import BaselineSystem
from .ContiAE import ContiAESystem
from .TransEmb import fscl_fastspeech2_class_factory
from .TransEmbADA import ada_class_factory, ssl_ada_class_factory
from .Seg import seg_fastspeech2_class_factory
from .Dual import dual_fastspeech2_class_factory
from .tune.FastSpeech2Tune import BaselineTuneSystem, fscl_tune_fastspeech2_class_factory
from .tune.DualTune import dual_tune_fastspeech2_class_factory

# Deprecated
# from .TransEmbOld import TransEmbSystem
from .TransEmbC import TransEmbCSystem
