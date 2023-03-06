from .FastSpeech2 import BaselineSystem
from .ContiAE import ContiAESystem
# from .TransEmbOrig import TransEmbOrigSystem, TransEmbOrig2System
from .TransEmb2 import TransEmbOrigSystem
from .TransEmb2 import TransEmbLinearSystem
from .TransEmbADA import ada_class_factory, ssl_ada_class_factory
from .tune.FastSpeech2Tune import BaselineTuneSystem, TransEmbOrigTuneSystem

# Deprecated
from .TransEmb import TransEmbSystem
from .TransEmbC import TransEmbCSystem
