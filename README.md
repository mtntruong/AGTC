# AGTC
Source code and data for the paper  
Attention-Guided Low-Rank Tensor Completion  
Truong Thanh Nhat Mai, Edmund Y. Lam, and Chul Lee  
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024  
https://doi.org/10.1109/TPAMI.2024.3429498

For PDF, please visit https://mtntruong.github.io/

If you have any question, please open an issue.  
The algorithm can also be applied to other applications. Please feel free to ask if you need help with training the algorithm using other datasets.

# Source code
The proposed algorithm is implemented in Python using PyTorch 1.11.  
We first upload the source codes of the proposed algorithm. Data and pre-trained weights will be updated later. Since the inputs of the proposed algorithm is as simple as
```
data   = torch.rand(1, 103, 64, 64)
omega  = torch.rand(1, 103, 64, 64) < 0.9
model  = RPCA_Net(N_iter=10)
output = model(data, omega)
```
you can easily plug this model into your training codes. N. B. The batch size must be 1, `omega` is binary, and the number of channels (103 in this example) is hard-coded in `main_net.py`.
Please also note that the source codes have not been refactored yet, so they are a little ugly.  
I will try to improve the readability and quality of this repository over time. I have been a bit busy recently due to company work.  
The training/testing scripts of AGTC is similar to those of [LRT-HDR](https://github.com/mtntruong/LRT-HDR). You may have a look at them in the meantime.

## Preparation

### Required Python packages
Please use `env.yml` to create an environment in [Anaconda](https://www.anaconda.com)
```
conda env create -f env.yml
```
Then activate the environment
```
conda activate agtc
```
If you want to change the environment name, edit the first line of `env.yml` before creating the environment.

### Data preprocessing
The `Data-Preparation` folder in each task contains datasets and source codes for preprocessing. The HSI datasets are included in the repository, while the HDR image datasets are uploaded to OneDrive (the links are provided in a text file). Note that the low dynamic range images in the HDR image datasets were already warped (aligned).

## Training
```
To be updated
```

# Citation
If our algorithm is useful for your research, please kindly cite our work
```
@ARTICLE{Mai2024,
  author={Mai, Truong Thanh Nhat and Lam, Edmund Y. and Lee, Chul},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Attention-Guided Low-Rank Tensor Completion}, 
  year={2024},
  pages={1-17},
  doi={10.1109/TPAMI.2024.3429498}}
}
