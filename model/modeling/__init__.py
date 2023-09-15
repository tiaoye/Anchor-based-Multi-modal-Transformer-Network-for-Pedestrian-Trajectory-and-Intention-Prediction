__all__ = ['build_model']

from .gtppo_tr import GTPPOTR

_MODELS_ = {
    'GtppoTR': GTPPOTR
}

def make_model(cfg):
    model = _MODELS_[cfg.METHOD]
    try:
        return model(cfg, dataset_name=cfg.DATASET.NAME)
    except:
        return model(cfg.MODEL, dataset_name=cfg.DATASET.NAME)
