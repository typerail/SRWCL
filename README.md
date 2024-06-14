## Requirements

python 3.9

torch 2.0.1

torchattackes 2.12.2

opencv-python 4.7.0.72


## Train:
python main.py --dataset train_mask --mode train_mask --image_dir './Datasets/train/' --image_val_dir './Datasets/val/'


## Embed:
python main.py --dataset test_embedding --mode test_embedding --image_dir './Datasets/test/'

## Test:
python main.py --dataset test_accuracy --mode test_accuracy --image_val_dir './results/embed/'

python main.py --dataset test_accuracy --mode test_accuracy --image_val_dir './results/capture/'



