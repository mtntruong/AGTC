## Preparation
### Download training/testing samples
- The dataset is already in the folder `Data-Preparation`.

- Run the script `./Data-Preparation/train_pairs_creator_PaviaU.m` to make training samples from the training set
  ```bash
  cd Data-Preparation/
  matlab -batch "train_pairs_creator_PaviaU"
  ```

### Pretrained weights
If you do not have time to retrain the network, you may use the pretrained weight stored in the folder `Weight`.

## Training
Start the training process by executing
```bash
python train.py --data_path=./Data-Preparation/Train_Pairs_PaviaU
```
Here is the tricky part, please cancel the training process every 20 epochs then rerun to update the learning rate using the following commands
```bash
# After 20th epoch
python train.py --data_path=./Data-Preparation/Train_Pairs_PaviaU --resume=./checkpoints/epoch_20.pth --set_lr=1e-6
# After 40th epoch
python train.py --data_path=./Data-Preparation/Train_Pairs_PaviaU --resume=./checkpoints/epoch_40.pth --set_lr=1e-7
# After 60th epoch
python train.py --data_path=./Data-Preparation/Train_Pairs_PaviaU --resume=./checkpoints/epoch_60.pth --set_lr=1e-8
# After 80th epoch
python train.py --data_path=./Data-Preparation/Train_Pairs_PaviaU --resume=./checkpoints/epoch_80.pth --set_lr=1e-9
# Stop after 100th epoch and you are done
```
After the training process completes, you should use the weight named `epoch_100.pth` for testing.

Regarding this weird practice, I have tried several ways to automatically update the learning rate during training, including `torch.optim.lr_scheduler`, but the performance is always worse (still better than competing algorithms). Manually adjusting the learning rate yields better results. 

## Testing
Generate the hyperspectral image from trained weights by executing
```bash
python test.py
```
The hyperspectral image `Pavia-AGTC.npy` will be written in the current folder.
