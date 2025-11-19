import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from careamics.lightning import TrainDataModule, VAEModule, create_predict_datamodule
from careamics.utils.metrics import scale_invariant_psnr
from careamics.prediction_utils import convert_outputs
from skimage.metrics import peak_signal_noise_ratio
from pytorch_lightning import Trainer


class PeriodicTestCallback(Callback):
    def __init__(
        self,
        test_observation: np.ndarray,
        test_signal: np.ndarray,
        train_data_module: TrainDataModule,
        test_interval: int = 10,
        batch_size: int = 64,
        tile_size: tuple = (128, 128),
        tile_overlap: tuple = (32, 32),
        mmse_counts: list[int] | int = [1],
    ):
        super().__init__()
        self.test_observation = test_observation
        self.test_signal = test_signal
        self.train_data_module = train_data_module
        self.test_interval = test_interval
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.mmse_counts = (
            [mmse_counts] if isinstance(mmse_counts, int) else mmse_counts
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.test_interval != 0:
            return

        means, stds = self.train_data_module.get_data_statistics()
        
        state = trainer.model.state_dict()
        model = VAEModule(pl_module.algorithm_config)
        model.load_state_dict(state)
        model.eval()

        pred_trainer = Trainer()

        for mmse_count in self.mmse_counts:
            model.algorithm_config.mmse_count = mmse_count
    
            # compute PSNR per-sample
            si_psnrs = np.zeros((len(self.test_observation),), dtype=np.float32)
            psnrs = np.zeros((len(self.test_observation),), dtype=np.float32)
            for i, (input, gt) in enumerate(zip(self.test_observation, self.test_signal)):
                pred_data_module = create_predict_datamodule(
                    pred_data=input,
                    data_type="array",
                    axes="YX",
                    batch_size=64,
                    tta_transforms=False,
                    image_means=means,
                    image_stds=stds,
                    tile_size=(128, 128),
                    tile_overlap=(32, 32),
                )
                pred = pred_trainer.predict(model, datamodule=pred_data_module)
                pred = convert_outputs(pred, tiled=True)[0]
                psnrs[i] = peak_signal_noise_ratio(gt, pred.squeeze(), data_range=255) 
                si_psnrs[i] = scale_invariant_psnr(gt, pred.squeeze())
            mean_psnr = float(psnrs.mean())
            std_psnr = float(psnrs.std())
            mean_si_psnr = float(si_psnrs.mean())
            std_si_psnr = float(si_psnrs.std())
            print(
                f"[epoch {epoch}] mmse={mmse_count}  PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} SI-PSNR: {mean_si_psnr:.2f} ± {std_si_psnr:.2f}"
            )

            metrics = {f"test/psnr_mmse_{mmse_count}": mean_psnr, f"test/si_psnr_mmse_{mmse_count}": mean_si_psnr}
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
