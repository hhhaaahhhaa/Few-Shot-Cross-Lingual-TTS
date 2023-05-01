import argparse
import os

import comet_ml
import pytorch_lightning as pl
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytorch_lightning.profiler import AdvancedProfiler
from Objects.config import LanguageDataConfigReader

from config.comet import COMET_CONFIG
from lightning.build import build_data_parsers, build_id2symbols
from lightning.datamodules import get_datamodule
from lightning.systems import get_system
import Define

quiet = False
if quiet:
    # NOTSET/DEBUG/INFO/WARNING/ERROR/CRITICAL
    os.environ["COMET_LOGGING_CONSOLE"] = "ERROR"
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    # configure logging at the root level of lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
if Define.CUDA_LAUNCH_BLOCKING:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


TRAINER_CONFIG = {
    "gpus": -1 if torch.cuda.is_available() else None,
    "strategy": "ddp" if torch.cuda.is_available() else None,
    "auto_select_gpus": True,
    "process_position": 1,
    "profiler": 'simple',
}


def main(args, configs):
    print("Prepare training ...")
    if Define.DEBUG:
        TRAINER_CONFIG.update({
            "limit_train_batches": 200,  # Useful for debugging
            "limit_val_batches": 50,  # Useful for debugging
        })

    data_configs, model_config, train_config, algorithm_config = configs

    #====== Parsing original format to general format (Deprecated) ======
    # data_configs = []
    # if Define.USE_OLD_CONFIG:
    #     for prep in preprocess_configs:
    #         parse_prep = {
    #             "name": prep["dataset"],
    #             "lang_id": prep["lang_id"],
    #             "unit_name": prep.get("unit_name", ""),
    #             "data_dir": prep["path"]["preprocessed_path"], 
    #             "subsets": prep["subsets"],
    #             "text_cleaners": prep["preprocessing"]["text"]["text_cleaners"], 
    #         }
    #         data_configs.append(parse_prep)

    # register parsers
    build_data_parsers(data_configs)

    # Check id2symbols mapping manually
    if Define.DEBUG:
        print(build_id2symbols(data_configs))

    # Determine to use frame-level/phoneme-level pitch and energy in FastSpeech2
    if "pitch" in model_config and "energy" in model_config:
        for data_config in data_configs:
            data_config["pitch"] = model_config["pitch"]
            data_config["energy"] = model_config["energy"]

    if Define.DEBUG:
        print("Initialize data parsers, done.")
    #==========================================================

    # for p in train_config["path"].values():
    #     os.makedirs(p, exist_ok=True)

    # Checkpoint for resume training or testing
    ckpt_file = None
    if args.exp_key is not None:
        ckpt_file = os.path.join(
            train_config["path"]["ckpt_path"], # COMET_CONFIG["project_name"],
            args.exp_key, args.ckpt_file
        )

    pretrain_ckpt_file = None
    if args.pretrain_path is not None:
        pretrain_ckpt_file = os.path.join(
            args.pretrain_path, args.ckpt_file
        )

    trainer_training_config = {
        'max_steps': train_config["step"]["total_step"],
        'log_every_n_steps': train_config["step"]["log_step"],
        'gradient_clip_val': train_config["optimizer"]["grad_clip_thresh"],
        'accumulate_grad_batches': train_config["optimizer"]["grad_acc_step"],
        'resume_from_checkpoint': ckpt_file,
    }
    if algorithm_config["type"] == 'imaml':
        # should manually clip grad
        del trainer_training_config['gradient_clip_val']
    
    if args.stage == 'train' or 'tune':
        # Init logger
        if Define.USE_COMET:
            comet_logger = pl.loggers.CometLogger(
                save_dir=train_config["path"]["log_path"],
                experiment_key=args.exp_key,
                experiment_name=args.exp_name,
                **COMET_CONFIG
            )
            comet_logger.log_hyperparams({
                "data_config": data_configs,
                "model_config": model_config,
                "train_config": train_config,
                "algorithm_config": algorithm_config,
            })
            loggers = [comet_logger]
            log_dir = os.path.join(comet_logger._save_dir, comet_logger.version)
            result_dir = os.path.join(
                train_config['path']['result_path'], comet_logger.version
            )
            ckpt_dir = os.path.join(
                train_config['path']['ckpt_path'], comet_logger.version
            )
        else:
            log_dir = f"{args.output_path}/log"
            result_dir = f"{args.output_path}/result"
            ckpt_dir = f"{args.output_path}/ckpt"
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)
            tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir)
            loggers = [tb_logger]
    else:
        assert args.exp_key is not None
        log_dir = os.path.join(
            train_config["path"]["log_path"], "meta", args.exp_key
        )
        if args.output_path is None:
            result_dir = os.path.join(
                train_config['path']['result_path'], args.exp_key
            )
        else:
            os.makedirs(args.output_path, exist_ok=True)
            result_dir = args.output_path

    # Get dataset
    # TODO: Refactor semi systems and non-semi systems in to same parsing process.
    # TODO: Separate data_config from preprocess_config.
    # if "semi" in algorithm_config["type"]:
    #     config_reader = LanguageDataConfigReader()
    #     data_configs = {
    #         # "sup": data_configs,
    #         "sup": [config_reader.read("_data/JSUT/4-shot/task-0")],
    #         "unsup": data_configs,
    #     }

    # TODO: need to enable data configs with reference when tuning with a better way
    tune_flag = args.tune
    if tune_flag:
        reference_data_configs = [data_configs[0]]
        # if len(data_configs) > 1:  # if multiple configs then separate the first config to be reference
        #     data_configs = data_configs[1:]
    datamodule = get_datamodule(algorithm_config["type"])(
        data_configs, model_config, train_config, algorithm_config, log_dir, result_dir
    )

    if Define.DEBUG:
        print("All components except system module are prepared.")
        input()
    if args.stage == 'train':
        system = get_system(algorithm_config["type"])
        # Load pretrained weights
        if pretrain_ckpt_file is None:
            model = system(
                data_configs, model_config, train_config, algorithm_config,
                log_dir, result_dir, ckpt_dir
            )
        else:
            print("Load from checkpoint...")
            model = system.load_from_checkpoint(
                pretrain_ckpt_file, 
                data_configs=data_configs, model_config=model_config, train_config=train_config, algorithm_config=algorithm_config,
                log_dir=log_dir, result_dir=result_dir, ckpt_dir=ckpt_dir
            )

        if Define.DEBUG:
            print("System module prepared.")
            input()

        # Train
        if Define.DEBUG:
            print("Start Training!")
        pl.seed_everything(43, True)
        trainer = pl.Trainer(logger=loggers, **TRAINER_CONFIG, **trainer_training_config)

        # Tune is viewed as tune_init + train
        if tune_flag:
            model.tune_init(reference_data_configs)
        trainer.fit(model, datamodule=datamodule)

    # TODO: Somewhat dirty, to be refactored
    # elif args.stage == 'test' or args.stage == 'predict':
    #     # Get model
    #     system = get_system(algorithm_config["type"])
    #     model = system.load_from_checkpoint(
    #         ckpt_file, 
    #         preprocess_config=preprocess_configs[0], model_config=model_config, train_config=train_config, algorithm_config=algorithm_config,
    #         log_dir=log_dir, result_dir=result_dir
    #     )
    #     # Test
    #     trainer = pl.Trainer(**TRAINER_CONFIG)
    #     trainer.test(model, datamodule=datamodule)

    # elif args.stage == 'debug':
    #     del datamodule
    #     datamodule = get_datamodule("base")(
    #         preprocess_configs, train_config, algorithm_config, log_dir, result_dir
    #     )
    #     datamodule.setup('test')
    #     for _ in tqdm(datamodule.test_dataset, desc="test_dataset"):
    #         pass

    # elif args.stage == 'tune':
    #     # Get model
    #     system = get_system(algorithm_config["type"])
    #     if pretrain_ckpt_file is not None:
    #         model = system.load_from_checkpoint(
    #             pretrain_ckpt_file,
    #             preprocess_config=preprocess_configs[0], model_config=model_config, train_config=train_config, algorithm_config=algorithm_config,
    #             log_dir=log_dir, result_dir=result_dir
    #         )
    #     else:
    #         model = system(
    #             preprocess_configs[0], model_config, train_config, algorithm_config,
    #             log_dir, result_dir
    #         )
    #     if "semi" in algorithm_config["type"]:
    #         # model.tune_init(data_configs["sup"][0])
    #         model.tune_init(config_reader.read("_data/JSUT/4-shot/task-0"))
    #     else:
    #         model.tune_init(data_configs[0])

    #     # Train
    #     if Define.USE_COMET:
    #         trainer = pl.Trainer(logger=loggers, **TRAINER_CONFIG, **trainer_training_config)
    #         pl.seed_everything(43, True)
    #     else:
    #         trainer = pl.Trainer(**TRAINER_CONFIG, **trainer_training_config)
    #     trainer.fit(model, datamodule=datamodule)
    #     trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocess_config", type=str, nargs='+', help="path to preprocess.yaml",
        default=['config/preprocess/miniLibriTTS.yaml'],
        # default=['config/preprocess/LibriTTS.yaml'],
    )
    parser.add_argument(
        "-m", "--model_config", type=str, help="path to model.yaml",
        default='config/model/dev.yaml',
        # default='config/model/base.yaml',
    )
    parser.add_argument(
        "-t", "--train_config", type=str, nargs='+', help="path to train.yaml",
        default=['config/train/dev.yaml', 'config/train/miniLibriTTS.yaml'],
        # default=['config/train/base.yaml', 'config/train/LibriTTS.yaml'],
    )
    parser.add_argument(
        "-a", "--algorithm_config", type=str, help="path to algorithm.yaml",
        default='config/algorithm/dev.yaml',
    )
    parser.add_argument(
        "-n", "--exp_name", type=str, help="experiment name, default is algorithm's name",
        default=None,
    )
    parser.add_argument(
        "-e", "--exp_key", type=str, help="experiment key",
        default=None,
    )
    parser.add_argument(
        "-pre", "--pretrain_path", type=str, help="pretrained model path",
        default=None,
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="output result path",
        default=None,
    )
    parser.add_argument(
        "-c", "--ckpt_file", type=str, help="ckpt file name",
        default="last.ckpt",
    )
    parser.add_argument(
        "-s", "--stage", type=str, help="stage (train/val/test/predict)",
        default="train",
    )
    # parser.add_argument(
    #     "-ie", "--index_exp", type=int, help="0-19",
    #     default=0,
    # )
    parser.add_argument(
        "-le", "--layer_exp", help="1-24", type=int,
        default=None,
    )
    parser.add_argument(
        "-ue", "--upstream_exp", type=str, help="upstream options",
        default="hubert_large_ll60k",
    )
    parser.add_argument("--use_comet", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--tune", action="store_true", default=False)
    parser.add_argument("--adapart", action="store_true", default=False)
    parser.add_argument("--nolid", action="store_true", default=False)
    parser.add_argument("--tunet2u", action="store_true", default=False)
    parser.add_argument("--atttemp", action="store_true", default=False)
    parser.add_argument("--conf", type=float, default=0.0)

    args = parser.parse_args()
    Define.DEBUG = args.debug
    Define.ADAPART = args.adapart
    Define.NOLID = args.nolid
    Define.TUNET2U = args.tunet2u
    Define.ATTTEMP = args.atttemp
    Define.PL_CONF = args.conf

    Define.USE_COMET = args.use_comet
    Define.LAYER_IDX = args.layer_exp
    Define.set_upstream(args.upstream_exp)
    print(f"Layer {args.layer_exp}, Upstream {args.upstream_exp}...")

    # Read Config. TODO: Tune and training currently are using different config format.
    if Define.USE_OLD_CONFIG:
        preprocess_configs = [
            yaml.load(open(path, "r"), Loader=yaml.FullLoader)
            for path in args.preprocess_config
        ]
    else:
        config_reader = LanguageDataConfigReader()
        preprocess_configs = [config_reader.read(path) for path in args.preprocess_config]
    
    model_config = yaml.load(
        open(args.model_config, "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(args.train_config[0], "r"), Loader=yaml.FullLoader
    )
    if len(args.train_config) > 1:
        train_config.update(
            yaml.load(open(args.train_config[1], "r"), Loader=yaml.FullLoader)
        )
    algorithm_config = yaml.load(
        open(args.algorithm_config, "r"), Loader=yaml.FullLoader
    )
    # algorithm_config["name"] += f"-{Define.UPSTREAM}-{Define.LAYER_IDX}"
    if args.exp_name is None:
        args.exp_name = algorithm_config["name"]
    if Define.UPSTREAM == "mel":
        Define.UPSTREAM_DIM = 80
        algorithm_config["adapt"]["phoneme_emb"]["representation_dim"] = 80
    configs = (preprocess_configs, model_config, train_config, algorithm_config)

    main(args, configs)