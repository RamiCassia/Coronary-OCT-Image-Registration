import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from branch_detection_solver import SolverBranchDetection
from branch_detection_data import OCTDataloader
from branch_detection_net import BranchDetector
from branch_detection_utils import load_checkpoint

parser = argparse.ArgumentParser(description="OCT Imaging - Branch Detection Training")
parser.add_argument(
    "--epochs", default=200, type=int, help="number of total epochs to run"
)
parser.add_argument("-b", "--batch-size", default=16, type=int)
parser.add_argument("--load", default=False)
parser.add_argument("--log-file", type=str, help="Log file name")
parser.add_argument("--checkpoint-file", type=str, help="Checkpoint file name")
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--gamma", default=0.99, type=float, help="LR scheduler gamma")
parser.add_argument("--step-size", default=1, type=int, help="LR scheduler step size")
parser.add_argument("--csv-path", type=str, help="Path to CSV file")
parser.add_argument("--img-path", type=str, help="Path to image files")
parser.add_argument("--experiment", type=str, help="Experiment name")
parser.add_argument("--start-epoch", type=int, default=0, help="Start epoch")


if __name__ == "__main__":
    DEVICE = torch.device("cuda")
    args = parser.parse_args()
    log_file = os.path.join(args.log_file, args.experiment)
    checkpoint_file = os.path.join(args.checkpoint_file, args.experiment)

    # Dataloading
    #transform = transforms.Compose(
    #    [transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees=180)]
    #)
    train_loader = OCTDataloader(
        split="train",
        batch_size=args.batch_size,
        img_path=args.img_path,
        csv_path=args.csv_path,
        device=DEVICE,
        transform=None,
    ).data_loader
    val_loader = OCTDataloader(
        split="val",
        batch_size=1,
        img_path=args.img_path,
        csv_path=args.csv_path,
        device=DEVICE,
        transform=None,
    ).data_loader
    test_loader = OCTDataloader(
        split="test",
        batch_size=1,
        img_path=args.img_path,
        csv_path=args.csv_path,
        device=DEVICE,
        transform=None,
    ).data_loader

    # loss_weight = torch.tensor([1., 4.], device=DEVICE)
    # branch_detection_loss = nn.CrossEntropyLoss(weight=loss_weight)
    branch_detection_loss = nn.CrossEntropyLoss()
    branch_detector = BranchDetector(num_blocks=1, spatial_res=1024)
    solver = SolverBranchDetection(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_function=branch_detection_loss,
        device=DEVICE,
        id_string=args.experiment,
        checkpoint_file=checkpoint_file,
        log_file=log_file,
    )
    branch_detector = branch_detector.to(DEVICE)

    optimiser = torch.optim.Adam(lr=args.lr, params=branch_detector.parameters())
    scheduler = optim.lr_scheduler.StepLR(
        optimiser, step_size=args.step_size, gamma=args.gamma
    )

    solver.train_network(
        model=branch_detector,
        optimiser=optimiser,
        scheduler=scheduler,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
    )

    branch_detector, _, _, _ = load_checkpoint(
        checkpoint=checkpoint_file,
        model=branch_detector,
        learning_rate=0,
        device=DEVICE,
    )
    solver.test_network(branch_detector)

