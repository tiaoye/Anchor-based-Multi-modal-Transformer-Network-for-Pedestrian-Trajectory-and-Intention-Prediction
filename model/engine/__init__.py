from model.engine.trainer import do_train 
from model.engine.trainer import do_val 
from model.engine.trainer import inference

ENGINE_ZOO = {
                'GtppoTR': (do_train, do_val, inference),
                }

def build_engine(cfg):
    return ENGINE_ZOO[cfg.METHOD]
