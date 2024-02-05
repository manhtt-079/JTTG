import logging
import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from utils import get_model_name


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('========')
    from evaluate import eval_picker, eval_writer, eval_jointmodel
    eval_writer(cfg, 'writer')

    # from train import train_picker, train_writer, train_jointmodel
    from evaluate import eval_picker, eval_writer, eval_jointmodel
    # from utils import MlflowWriter

    # mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')

    # if cfg.model.name == 'picker':
    #     experiment_name = cfg.dataset.name
    #     run_name = get_model_name(cfg)
    #     writer = MlflowWriter(experiment_name, run_name)
    #     writer.log_params_from_omegaconf_dict(cfg)
    #     train_picker(cfg, writer)
    #     eval_picker(cfg, writer)

    # elif cfg.model.name == 'writer':
    #     experiment_name = cfg.dataset.name
    #     run_name = get_model_name(cfg)
    #     writer = MlflowWriter(experiment_name, run_name)
    #     writer.log_params_from_omegaconf_dict(cfg)
    #     train_writer(cfg, writer)
    #     eval_writer(cfg, writer)

    # elif cfg.model.name == 'jointmodel':
    #     experiment_name = cfg.dataset.name
    #     run_name = get_model_name(cfg)
    #     writer = MlflowWriter(experiment_name, run_name)
    #     writer.log_params_from_omegaconf_dict(cfg)
    #     train_jointmodel(cfg, writer)
    #     eval_jointmodel(cfg, writer)


if __name__ == "__main__":
    main()
