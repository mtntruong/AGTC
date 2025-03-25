## Preparation
### Download training/testing samples
- The dataset is already in the folder `Data-Preparation`.

- Run the script `./Data-Preparation/train_pairs_creator_Landsat.m` to make training samples from the training set
  ```bash
  cd Data-Preparation/
  matlab -batch "train_pairs_creator_Landsat"
  ```

### Pretrained weights
If you do not have time to retrain the network, you may use the pretrained weight stored in the folder `Weight`.

## Training
Start the training process by executing
```bash
python train.py --data_path=./Data-Preparation/Train_Pairs_Landsat
```
Here is the tricky part, please cancel the training process every 10 epochs then rerun to update the learning rate using the following commands
```bash
# After 10th epoch
python train.py --data_path=./Data-Preparation/Train_Pairs_Landsat --resume=./checkpoints/epoch_10.pth --set_lr=1e-6
# After 20th epoch
python train.py --data_path=./Data-Preparation/Train_Pairs_Landsat --resume=./checkpoints/epoch_20.pth --set_lr=1e-7
# After 30th epoch
python train.py --data_path=./Data-Preparation/Train_Pairs_Landsat --resume=./checkpoints/epoch_30.pth --set_lr=1e-8
# After 40th epoch
python train.py --data_path=./Data-Preparation/Train_Pairs_Landsat --resume=./checkpoints/epoch_40.pth --set_lr=1e-9
# Stop after 50th epoch and you are done
```
After the training process completes, you should use the weight named `epoch_50.pth` for testing.

Regarding this weird practice, I have tried several ways to automatically update the learning rate during training, including `torch.optim.lr_scheduler`, but the performance is always worse (still better than competing algorithms). Manually adjusting the learning rate yields better results. 

## Testing
Generate the HDR images from trained weights by executing
```bash
python test.py
```
The synthesized HDR images will be written in the folder specified by `--output-path`
