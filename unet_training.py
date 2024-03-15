import logging
import sys
import tempfile
import torch
from ignite.engine import Events, _prepare_batch, create_supervised_trainer
from ignite.handlers import ModelCheckpoint

import monai
from monai.handlers import StatsHandler
from data import setup_data


def main(tempdir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # prepare data loader
    data_dir = '/data/to_huairuo/ski10train/train_nii'
    train_loader = setup_data(data_dir)

    # create UNet, DiceLoss and Adam optimizer
    # out_channels = num_class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss = monai.losses.DiceLoss(softmax=True, to_onehot_y=True)
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)

    # Ignite trainer expects batch=(img, seg) and returns output=loss at every iteration
    # can add output_transform to return other values
    def prepare_batch(batch, device=None, non_blocking=False):
        return _prepare_batch((batch["img"], batch["seg"]), device, non_blocking)

    trainer = create_supervised_trainer(net, opt, loss, device, False, prepare_batch=prepare_batch)

    # adding checkpoint handler to save models (network params and optimizer stats) during training
    # n_saved best models will be saved in ./runs_dict
    checkpoint_handler = ModelCheckpoint("./runs_dict/", "net", n_saved=2, require_empty=False)
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={"net": net, "opt": opt},
    )

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # no metrics for trainer here, just loss
    # can use output_transform to convert engine.state.output if it's not loss value
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    # define number of epochs to train
    train_epochs = 20
    state = trainer.run(train_loader, train_epochs)
    print(state)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
