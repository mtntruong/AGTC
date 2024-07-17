import os
import torch
import argparse

from util import MainNetDataset
from tqdm import tqdm
from torchinfo import summary
from main_net import RPCA_Net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        default='hdr_ckpts',
        help='Path for checkpointing.',
    )
    parser.add_argument(
        '--resume',
        help='Resume training from saved checkpoint(s).',
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=1,
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
        '--set_lr',
        type=float,
        default=-1,
        help='Set new learning rate.',
    )
    return parser.parse_args()


def train(opt):

    torch.backends.cudnn.benchmark = True

    train_path = '../../HDR_DATA/TRAIN/HDM_warped_ssim'
    data_train = MainNetDataset(train_path)
    data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, num_workers=4)

    model = RPCA_Net(N_iter=opt.N_iter)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    loss = torch.nn.L1Loss()
    summary(model, input_size=[(1, 3, 128, 128), (1, 3, 128, 128), (1, 3, 128, 128),\
                               (1, 3, 128, 128), (1, 3, 128, 128), (1, 3, 128, 128)])

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
    for epoch in tqdm(range(epoch_0, 51), desc='Training'):
        for i, (img0, img1, img2, omg0, omg1, omg2, label) in enumerate(tqdm(data_train_loader, desc=f'Epoch {epoch}')):

            img0 = img0.cuda()
            img1 = img1.cuda()
            img2 = img2.cuda()
            omg0 = omg0.cuda()
            omg1 = omg1.cuda()
            omg2 = omg2.cuda()
            label = label.cuda()

            X_hat, X_hdr = model(img0, img1, img2, omg0, omg1, omg2)
            total_loss = loss(X_hat, torch.cat((label, label, label), dim=1)) + loss(X_hdr, label)

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
