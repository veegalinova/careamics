from pathlib import Path

import tifffile
import numpy as np

import urllib 
import os
import zipfile

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from careamics.lightning.callbacks import HyperParametersCallback
from pytorch_lightning.loggers import WandbLogger

from careamics.config import create_hdn_configuration
from careamics.lightning import TrainDataModule
from careamics.lightning.lightning_module import VAEModule

from careamics.prediction_utils import convert_outputs
from careamics.lightning import create_predict_datamodule
from careamics.utils.metrics import scale_invariant_psnr
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel 
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig

from pytorch_lightning import seed_everything

from callbacks import PeriodicTestCallback 


def train_noise_model(data_path):
    observation = tifffile.imread(data_path + '20190726_tl_50um_500msec_wf_130EM_FD.tif') 
    signal = np.mean(observation[:, ...],axis=0)[np.newaxis,...]
    min_signal=np.min(signal)
    max_signal=np.max(signal)
    print("Minimum Signal Intensity is", min_signal)
    print("Maximum Signal Intensity is", max_signal)

    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        min_signal=min_signal,
        max_signal=max_signal,
        n_gaussian=3,
        n_coeff=2,
        min_sigma=50
    )

    gaussianMixtureNoiseModel = GaussianMixtureNoiseModel(nm_config)

    gaussianMixtureNoiseModel.fit(
        signal, 
        observation, 
        batch_size = 250000, 
        n_epochs = 2000,
        learning_rate=0.1
    )
    gaussianMixtureNoiseModel.save("./", "noise_model")

    nm_config.path = "./noise_model.npz"

    return nm_config


def main():
    seed_everything(42)

    zipPath="./data/Convallaria_diaphragm.zip"

    if not os.path.exists(zipPath): 
        Path('./data').mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlretrieve('https://zenodo.org/record/5156913/files/Convallaria_diaphragm.zip?download=1', zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall("./data")

    path = "./data/Convallaria_diaphragm/"

    nm_config = train_noise_model(path)
    
    observation = tifffile.imread(path+'20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif')
    train_data = observation[:int(0.85*observation.shape[0])]
    val_data= observation[int(0.85*observation.shape[0]):]

    print(
        "Shape of training images:", train_data.shape, 
        "Shape of validation images:", val_data.shape
    )

    experiment_name="val_run"
    project = "careamics_hdn"
    root = Path(
        f"/group/jug/Vera/projects/careamics/dev/HDN_debug/experiments/logs/{experiment_name}/"
    )

    config = create_hdn_configuration(
        experiment_name=experiment_name,
        data_type="array",
        axes="SYX",
        patch_size=(64, 64),
        batch_size=16,
        num_epochs=400,
        predict_logvar="pixelwise"
    )
    config.data_config.seed = 42

    # Network params from original paper
    # Data 
    config.data_config.patch_size = (64, 64)
    config.data_config.batch_size = 16

    # Training 
    config.algorithm_config.num_epochs = 400  # with reduced steps per epoch! 
    config.algorithm_config.optimizer.parameters = {'lr': 3e-4}
    config.algorithm_config.loss.kl_params.free_bits_coeff = 1.0
    config.algorithm_config.lr_scheduler.parameters = {'patience': 10, 'factor': 0.5, 'min_lr': 1e-12}

    # Model 
    config.algorithm_config.model.encoder_n_filters = 64
    config.algorithm_config.model.decoder_n_filters = 64
    # config.algorithm_config.model.z_dims = [32]*6
    config.algorithm_config.model.encoder_dropout = 0.2
    config.algorithm_config.model.decoder_dropout = 0.2
    config.algorithm_config.model.nonlinearity = "ELU"
    config.algorithm_config.noise_model = MultiChannelNMConfig(noise_models=[nm_config])

    # hardcoded or does not work, need to expose parameters
    model = VAEModule(config.algorithm_config)
    model.model.encoder_blocks_per_layer = 5
    model.model.decoder_blocks_per_layer = 5
    model.algorithm_config.mmse_count = 100

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params}")

    observation= tifffile.imread(path + '20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif')
    observation = observation.astype(np.float32)[:,:512,:512]
    signal = np.mean(observation[:,...],axis=0)[np.newaxis,...]

    train_data_module = TrainDataModule(
        data_config=config.data_config,
        train_data=train_data,
        val_data=observation
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=root / Path("checkpoints"),
            filename=f"{config.experiment_name}_{{epoch:02d}}_step_{{step}}",
            **config.training_config.checkpoint_callback.model_dump()
        ),
        HyperParametersCallback(config),
        PeriodicTestCallback(
            test_observation=observation,
            test_signal=np.repeat(signal, len(observation), axis=0),
            train_data_module=train_data_module,
            test_interval=1,
            batch_size=2,
            tile_size=(128, 128),
            tile_overlap=(32, 32),
            mmse_counts=[1]
        ),
    ]
    trainer = Trainer(
        max_epochs=config.algorithm_config.num_epochs,   
        limit_train_batches=500, 
        default_root_dir=root,
        logger=WandbLogger(
            project=project, 
            name=experiment_name, 
            config=config.model_dump(),
            save_dir=root / Path("wandb_logs"),
        ),
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=train_data_module)

    # Final test evaluation (optional - already done periodically during training)
    # The test data is just one quater of the full image ([:,:512,:512]) following the works which have used this data earlier

    means, stds = train_data_module.get_data_statistics()
    pred_data_module = create_predict_datamodule(
        pred_data=observation,
        data_type="array",
        axes="SYX",
        batch_size=2,
        tta_transforms=False,
        image_means=means,
        image_stds=stds,
        tile_size=(128, 128),
        tile_overlap=(32, 32),
    )
    prediction = trainer.predict(model, datamodule=pred_data_module)
    prediction = convert_outputs(prediction, tiled=True)

    noises = observation
    gts = np.repeat(signal, len(noises), axis=0)

    psnrs = np.zeros((len(prediction), 1))

    for i, (pred, gt) in enumerate(zip(prediction, gts)):
        psnrs[i] = scale_invariant_psnr(gt, pred.squeeze())

    test_psnr_mean = psnrs.mean()
    print(f"Final Test PSNR: {test_psnr_mean:.2f} +/- {psnrs.std():.2f}")
    print("Reported PSNR: 37.39")

    trainer.logger.experiment.log({"test_psnr": test_psnr_mean})

if __name__ == "__main__":
    main()
