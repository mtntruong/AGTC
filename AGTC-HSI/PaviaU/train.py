import os
import torch
import argparse

from util import HSIDataset
from tqdm import tqdm
from torchinfo import summary
from main_net import RPCA_Net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        required=True,
        help='Path to training data.',
    )
    parser.add_argument(
        '--save_path',
        default='checkpoints',
        help='Path for checkpointing.',
    )
    parser.add_argument(
        '--resume',
        help='Resume training from saved checkpoint(s).',
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=2,
        help='Checkpoint model every x epochs.',
    )
    parser.add_argument(
        '--loss_freq',
        type=int,
        default=20,
        help='Report (average) loss every x iterations.',
    )
    parser.add_argument(
        '--N_iter',
        type=int,
        default=10,
        help='Number of unrolled iterations.',
    )
    parser.add_argument(
        '--input_dim',
        type=int,
        default=103,
        help='Number of channels of the input tensor.',
    )
    parser.add_argument(
        '--set_lr',
        type=float,
        default=-1,
        help='Set new learning rate.',
    )
    return parser.parse_args()


def train(opt):

    torch.backends.cudnn.benchmark = True

    train_path = opt.data_path
    data_train = HSIDataset(train_path)
    data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, num_workers=4)

    model = RPCA_Net(N_iter=opt.N_iter, tensor_num_channels=opt.input_dim)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    loss = torch.nn.L1Loss()
    summary(model, input_size=[(1, opt.input_dim, 64, 64), (1, opt.input_dim, 64, 64)])

    if opt.resume is not None:
        print('Resume training from' + opt.resume)
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_0 = checkpoint['epoch'] + 1
        model.train()
    else:
        print('Start training from scratch.')
        epoch_0 = 1

    if not opt.set_lr == -1:
        for groups in optimizer.param_groups: groups['lr'] = opt.set_lr; break
        print('New learning rate:', end=" ")
        for groups in optimizer.param_groups: print(groups['lr']); break

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    else:
        print('WARNING: save_path already exists. Checkpoints may be overwritten.')

    avg_loss = 0
    for epoch in tqdm(range(epoch_0, 101), desc='Training'):
        for i, (data, omega, target) in enumerate(tqdm(data_train_loader, desc=f'Epoch {epoch}')):

            data, omega, target = data.cuda(), omega.cuda(), target.cuda()

            img = model(data, omega)
            total_loss = loss(img, target)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            avg_loss += total_loss.item()
            if ((i + 1) % opt.loss_freq) == 0:
                rep = (
                    f'Epoch: {epoch:>5d}, '
                    f'Iter: {i+1:>6d}, '
                    f'Loss: {avg_loss/opt.loss_freq:>6.2e}'
                )
                tqdm.write(rep)
                avg_loss = 0

        if (epoch % opt.checkpoint_freq) == 0:
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(opt.save_path, f'epoch_{epoch}.pth')
            )


if __name__ == '__main__':
    opt_args = parse_args()
    train(opt_args)
