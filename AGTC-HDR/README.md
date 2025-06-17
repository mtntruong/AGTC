## Preparation
### Download training/testing samples
- Download the following files 
  - Training set: [LINK](https://dguackr-my.sharepoint.com/:u:/g/personal/mtntruong_dgu_ac_kr/EZzDojamk4dNh_fnp10CqAkBpesaufsu-cQ075YZ2V04-Q)
  - Test set: [LINK](https://dguackr-my.sharepoint.com/:u:/g/personal/mtntruong_dgu_ac_kr/EQ5DyEDwkcpDmt29te_OrXIBayM86RNBudegf_K_CFDXpQ)

  then put them in the folder `Data-Preparation` and unzip.

- Run the script `./Data-Preparation/data_prep.py` to make training samples from the training set
  ```bash
  cd Data-Preparation/
  python data_prep.py
  ```

### Pretrained weights
If you do not have time to retrain the network, you may use the pretrained weight stored in the folder `Weight`.

## Training
Start the training process by executing
```bash
python train.py --data_path=./Data-Preparation/IMG_Patches
```
Here is the tricky part, please cancel the training process every 10 epochs then rerun to update the learning rate using the following commands
```bash
# After 10th epoch
python train.py --data_path=./Data-Preparation/IMG_Patches --resume=./checkpoints/epoch_10.pth --set_lr=1e-6
# After 20th epoch
python train.py --data_path=./Data-Preparation/IMG_Patches --resume=./checkpoints/epoch_20.pth --set_lr=1e-7
# After 30th epoch
python train.py --data_path=./Data-Preparation/IMG_Patches --resume=./checkpoints/epoch_30.pth --set_lr=1e-8
# After 40th epoch
python train.py --data_path=./Data-Preparation/IMG_Patches --resume=./checkpoints/epoch_40.pth --set_lr=1e-9
# Stop after 50th epoch and you are done
```
After the training process completes, you should use the weight named `epoch_50.pth` for testing.

Regarding this weird practice, I have tried several ways to automatically update the learning rate during training, including `torch.optim.lr_scheduler`, but the performance is always worse (still better than competing algorithms). Manually adjusting the learning rate yields better results. 

## Testing
Generate the HDR images from trained weights by executing
```bash
python test.py --data_path=./Data-Preparation/HDR-Test --output_path=./HDR_Outputs --checkpoint=./Weight/AGTC-HDR.pth
```
The synthesized HDR images will be written in the folder specified by `--output-path`
