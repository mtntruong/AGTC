# AGTC
Source code and data for the paper  
Attention-Guided Low-Rank Tensor Completion  
Truong Thanh Nhat Mai, Edmund Y. Lam, and Chul Lee  
IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 9818-9833, 2024  
https://doi.org/10.1109/TPAMI.2024.3429498

For PDF, please visit https://mtntruong.github.io/  
The appendix, which is somehow not included in the publisher's version, is freely available on the above website or can be directly accessed at [this link](https://mtntruong.github.io/assets/pdf/2024_TPAMI_supp.pdf).

If you have any questions, please open an issue.  
The algorithm can also be applied to other applications. Please feel free to ask if you need help training the algorithm on other datasets.

# Source code
The proposed algorithm is implemented in Python using PyTorch 1.11. There are two subfolders in this repository:
- AGTC-HDR: application of the proposed algorithm in multi-exposure fusion-based HDR imaging (Section 4.1 in the paper)
- AGTC-HSI: application in hyperspectral image restoration (Section 4.2 in the paper)

Since the input of the proposed deep network is as simple as
```python
data   = torch.rand(1, 103, 64, 64).cuda()
omega  = (torch.rand(1, 103, 64, 64) < 0.9).float().cuda()
model  = RPCA_Net(N_iter=10, tensor_num_channels=103).cuda()
output = model(data, omega)
```
you can easily plug this model into your training codes. Important notes:
- The batch size must be 1.
- The variable `omega` is binary, since it's a mask indicating observed entries.

## Preparation
Please use `env.yml` to create an environment with [Anaconda](https://www.anaconda.com)
```
conda env create -f env.yml
```
Then activate the environment
```
conda activate agtc
```
If you want to change the environment name, edit the first line of `env.yml` before creating the environment.

## Data preprocessing, training, and testing
Please see the `README.md` file in each subfolder for detailed instructions on downloading and preprocessing data, running scripts, and using pre-trained weights for the corresponding task.

# Citation
If our algorithm is useful for your research, please kindly cite our work
```
@ARTICLE{Mai2024,
  author={Mai, Truong Thanh Nhat and Lam, Edmund Y. and Lee, Chul},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Attention-Guided Low-Rank Tensor Completion}, 
  year={2024},
  volume={46},
  number={12},
  pages={9818-9833},
  doi={10.1109/TPAMI.2024.3429498}
}
```
