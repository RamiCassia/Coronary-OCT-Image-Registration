import torch
import argparse
from BranchDetection.branch_detection_net import BranchDetector
from BranchDetection.branch_detection_utils import load_checkpoint, predict_sidebranches

parser = argparse.ArgumentParser(description="OCT Imaging - Pipeline")
parser.add_argument("--checkpoint-file", type=str, help="Checkpoint file name")
parser.add_argument("--img-path", type=str, help="Img path")
parser.add_argument("--img-name", type=str, help="Img name")

# To run the code with the current branch_detector checkoint use:
# python pipeline.py --checkpoint-file "/store/DAMTP/cr661/OCTImaging/checkpoints/lr0001" 
# # --img-path "/store/DAMTP/cr661/OCTImaging/data/5_Annotated_Files/SplittedImages" 
# # --img-name "IBIS4_453-636-202_LAD_NI_B_BL_2.png" (change accordingly)
if __name__ == "__main__":
    args = parser.parse_args()

    # Branch detection
    DEVICE = torch.device("cuda")
    branch_detector = BranchDetector(num_blocks=1, spatial_res=1024).to(DEVICE)
    branch_detector, _, _, _ = load_checkpoint(
        checkpoint=args.checkpoint_file,
        model=branch_detector,
        learning_rate=0,
        device=DEVICE,
    )
    # Prediction of sidebranches via trained neural network
    idx, slide_num, prediction = predict_sidebranches(
        branch_detector=branch_detector,
        img_path=args.img_path,
        img_name=args.img_name,
        device=DEVICE,
    )
