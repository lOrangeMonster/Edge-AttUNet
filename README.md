# Edge-AttUNet
An enhanced U-Net reconstruction model that converts 8-bit low dynamic range (LDR) fluorescence images into high-fidelity 16-bit high dynamic range (HDR) images under low-light conditions.
## Configuration
pip install -r requirements.txt
## How to test
cd codes
python test.py -opt options/test/test.yml
## How to train
cd codes
python train.py -opt options/train/train.yml
The models and training states will be saved to ./experiments/name.
## Acknowledgment
The code is inspired by [HDRUNet](https://github.com/chxy95/HDRUNet)
