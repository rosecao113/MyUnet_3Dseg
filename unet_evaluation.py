import logging
import sys
import tempfile
import torch
from ignite.engine import Engine

import monai
from monai.data import decollate_batch
from monai.handlers import CheckpointLoader, MeanDice, StatsHandler
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    SaveImage,
)
from data import setup_data


def main(tempdir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # create same Unet as training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # sliding window size and batch size for windows inference
    roi_size = (64, 64, 64)
    sw_batch_size = 4

    # post transforms to generate mask
    post_trans = Compose([AsDiscrete(argmax=True)])
    # save mask to ./tempdir
    save_image = SaveImage(output_dir="tempdir", output_ext=".nii.gz", output_postfix="seg")

    def _sliding_window_processor(engine, batch):
        net.eval()
        with torch.no_grad():
            val_images, val_labels = batch["img"].to(device), batch["seg"].to(device)
            seg_probs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
            seg_probs = [post_trans(i) for i in decollate_batch(seg_probs)]
            for seg_prob in seg_probs:
                save_image(seg_prob)
            return seg_probs, val_labels

    evaluator = Engine(_sliding_window_processor)

    # evaluation metric
    MeanDice().attach(evaluator, "Mean_Dice")

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # no need to print loss for evaluator, just metrics
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss, so disable per iteration output
    )
    val_stats_handler.attach(evaluator)

    # load the model trained by "unet_training"
    CheckpointLoader(load_path="./runs_dict/net_checkpoint_600.pt", load_dict={"net": net}).attach(evaluator)

    # sliding window inference for one image at every iteration
    val_loader = setup_data(data_dir='/data/to_huairuo/ski10train/val_nii', train=False)
    state = evaluator.run(val_loader)
    print(state)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
