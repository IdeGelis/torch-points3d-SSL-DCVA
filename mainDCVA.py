import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.mainDCVA import DCVA
import shutil
import os.path as osp

@hydra.main(config_path="conf", config_name="configDCVA_3D") #evalUrb3D configDCVA_3D
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    dcva = DCVA(cfg)
    dcva.main_DCVA()
    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
