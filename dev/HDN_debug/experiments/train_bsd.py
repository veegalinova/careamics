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
from skimage.metrics import peak_signal_noise_ratio

from careamics.config import create_hdn_configuration
from careamics.lightning import TrainDataModule
from careamics.lightning.lightning_module import VAEModule

from careamics.prediction_utils import convert_outputs
from careamics.lightning import create_predict_datamodule
from careamics.utils.metrics import scale_invariant_psnr

from pytorch_lightning import seed_everything

from callbacks import PeriodicTestCallback 


def main():
    seed_everything(42)

    zipPath="data/BSD68_reproducibility.zip"

    if not os.path.exists(zipPath): 
        Path('./data').mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlretrieve('https://cloud.mpi-cbg.de/index.php/s/pbj89sV6n6SyM29/download', zipPath)
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall("./data")

    train_data = np.load('data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25.npy')
    val_data = np.load('data/BSD68_reproducibility_data/val/DCNN400_validation_gaussian25.npy')

    print(
        "Shape of training images:", train_data.shape, 
        "Shape of validation images:", val_data.shape
    )

    experiment_name="careamics_adamax_nologvar_noscaling"
    project = "careamics_hdn"
    root = Path(
        f"/group/jug/Vera/projects/careamics/dev/HDN_debug/experiments/logs/{experiment_name}/"
    )

    config = create_hdn_configuration(
        experiment_name=experiment_name,
        data_type="array",
        axes="SYX",
        patch_size=(128, 128),
        batch_size=16,
        num_epochs=400,
        predict_logvar=None
    )
    config.data_config.seed = 42

    # Network params from original paper

    # Training 
    config.algorithm_config.num_epochs = 400  # with reduced steps per epoch! 
    config.algorithm_config.optimizer.parameters = {'lr': 3e-4}
    config.algorithm_config.loss.kl_params.free_bits_coeff = 1.0
    config.algorithm_config.lr_scheduler.parameters = {'patience': 10, 'factor': 0.5, 'min_lr': 1e-12}

    # Model 
    config.algorithm_config.model.encoder_n_filters = 64
    config.algorithm_config.model.decoder_n_filters = 64
    config.algorithm_config.model.z_dims = [32]*6
    config.algorithm_config.model.encoder_dropout = 0.2
    config.algorithm_config.model.decoder_dropout = 0.2
    config.algorithm_config.model.nonlinearity = "ELU"

    # hardcoded or does not work, need to expose parameters
    model = VAEModule(config.algorithm_config)
    model.model.encoder_blocks_per_layer = 5
    model.model.decoder_blocks_per_layer = 5
    model.algorithm_config.mmse_count = 100

    # config.algorithm_config.loss.reconstruction_weight = 1 / ((25 / 63.352882) ** 2)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params}")

    train_data_module = TrainDataModule(
        data_config=config.data_config,
        train_data=train_data,
        val_data=val_data
    )

    test_images = np.load('data/BSD68_reproducibility_data/test/bsd68_gaussian25.npy', allow_pickle=True)[:10]
    test_images_gt = np.load('data/BSD68_reproducibility_data/test/bsd68_groundtruth.npy', allow_pickle=True)[:10]

    callbacks = [
        ModelCheckpoint(
            dirpath=root / Path("checkpoints"),
            filename=f"{config.experiment_name}_{{epoch:02d}}_step_{{step}}",
            **config.training_config.checkpoint_callback.model_dump()
        ),
        HyperParametersCallback(config),
        PeriodicTestCallback(
            test_observation=test_images,
            test_signal=test_images_gt,
            train_data_module=train_data_module,
            test_interval=5,
            batch_size=16,
            tile_size=(128, 128),
            tile_overlap=(32, 32),
            mmse_counts=[1, 15],
        ),
    ]
    trainer = Trainer(
        max_epochs=config.algorithm_config.num_epochs,   
        limit_train_batches=500,  
        default_root_dir=root,
        logger=WandbLogger(
            project=project, 
            name=experiment_name, 
            group="ablations",
            config=config.model_dump(),
            save_dir=root / Path("wandb_logs"),
        ),
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=train_data_module)

    
    test_images = np.load('data/BSD68_reproducibility_data/test/bsd68_gaussian25.npy', allow_pickle=True)
    test_images_gt = np.load('data/BSD68_reproducibility_data/test/bsd68_groundtruth.npy', allow_pickle=True)

    print(f"Shape of test images: {test_images.shape}")
    print(f"Shape of test images GT: {test_images_gt.shape}")

    si_psnrs = np.zeros((len(test_images), 1))
    psnrs = np.zeros((len(test_images), 1))

    means, stds = train_data_module.get_data_statistics()
    for i, (test_image, test_gt) in enumerate(zip(test_images, test_images_gt)):
        pred_data_module = create_predict_datamodule(
            pred_data=test_image,
            data_type="array",
            axes="YX",
            batch_size=64,
            tta_transforms=False,
            image_means=means,
            image_stds=stds,
            tile_size=(128, 128),
            tile_overlap=(32, 32),
        )
        prediction = trainer.predict(model, datamodule=pred_data_module)
        prediction = convert_outputs(prediction, tiled=True)[0]

        si_psnrs[i] = scale_invariant_psnr(test_gt, prediction.squeeze())
        psnrs[i] = peak_signal_noise_ratio(test_gt, prediction.squeeze(), data_range=255)

    si_psnr_mean = si_psnrs.mean()
    test_psnr_mean = psnrs.mean()
    print(f"SI-PSNR: {si_psnr_mean:.2f} +/- {si_psnrs.std():.2f}")
    print(f"PSNR: {test_psnr_mean:.2f} +/- {psnrs.std():.2f}")
    print("Reported PSNR: 28.82")

    trainer.logger.experiment.log({"test_psnr": si_psnr_mean})
    trainer.logger.experiment.log({"test_psnr_debug": test_psnr_mean})

if __name__ == "__main__":
    main()