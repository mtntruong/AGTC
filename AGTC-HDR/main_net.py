import torch
from torch import nn


class SpatialAttentionModule(nn.Module):
    def __init__(self, n_feats):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(x))))
        return att_map


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        return x


class RPCA_Block(nn.Module):
    def __init__(self):
        super(RPCA_Block, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB(growRate0=64, growRate=32, nConvLayers=8)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, C_k):
        C = torch.fft.fft(torch.squeeze(C_k), n=C_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(C, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), C).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)


    def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C):

        # Update C
        psi_c = self.mu + self.alpha
        Psi_C = (L1 - L2 + self.mu * Omega - self.mu * E - self.mu * T + self.alpha * P)
        C_k = torch.div(torch.mul(W, self.tensor_product(L, R)) + Psi_C, W + psi_c)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, C_k)

        # Update E
        psi_e = self.mu + self.beta
        Psi_E = (L1 - L3 + self.mu * Omega - self.mu * C_k - self.mu * T + self.beta * Q) / psi_e
        E_k = torch.mul(torch.sign(Psi_E), nn.functional.relu(torch.abs(Psi_E) - self.lamb / psi_e))

        # Update T
        Y = Omega - C_k - E_k + L1 / self.mu
        T_k = torch.mul(Y, Omega_C) + \
              torch.mul(Y, Omega) * torch.min(torch.tensor(1.).cuda(), \
                        self.delta / (torch.norm(torch.mul(Y, Omega), 'fro') + 1e-6))

        # Update P
        P_k = self.Proximal_P(C_k + L2 / (self.alpha + 1e-6))

        # Update Q
        Q_k = self.Proximal_Q(E_k + L3 / (self.beta + 1e-6))

        # Update Lambda
        L1_k = L1 + self.mu * (Omega - C_k - E_k - T_k)
        L2_k = L2 + self.alpha * (C_k - P_k)
        L3_k = L3 + self.beta * (E_k - Q_k)

        return L_k, R_k, C_k, E_k, T_k, P_k, Q_k, L1_k, L2_k, L3_k


class RPCA_Net(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(9)

        # Finale weighted sum
        self.composer = nn.Conv2d(9, 3, kernel_size=1, stride=1, bias=False)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, img0, img1, img2, omg0, omg1, omg2):

        # Observation
        OmegaD = torch.cat((img0, img1, img2), dim=1)
        Omega_C = torch.tensor(1.).cuda() -  torch.cat((omg0, omg1, omg2), dim=1)

        # Weight W for reweigted least-squares
        W = self.att_module(OmegaD)

        # Optimal variables
        OmegaD = torch.mul(OmegaD, torch.cat((omg0, omg1, omg2), dim=1))
        C  = OmegaD
        L1 = torch.zeros(C.size(), device=torch.device('cuda'))
        L2 = torch.zeros(C.size(), device=torch.device('cuda'))
        L3 = torch.zeros(C.size(), device=torch.device('cuda'))
        E  = torch.zeros(C.size(), device=torch.device('cuda'))
        T  = torch.zeros(C.size(), device=torch.device('cuda'))
        P  = torch.zeros(C.size(), device=torch.device('cuda'))
        Q  = torch.zeros(C.size(), device=torch.device('cuda'))
        # Init L/R
        L = torch.ones((9, 3, OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((3, OmegaD.shape[-2], OmegaD.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, C, E, T, P, Q, L1, L2, L3] = self.network[i](L, R, C, E, T, P, Q, L1, L2, L3, OmegaD, W, Omega_C)
            layers.append([L, R, C, E, T, P, Q, L1, L2, L3])

        # Synthesize
        X_hat = layers[-1][2]
        X_hdr = self.composer(X_hat)

        return X_hat, X_hdr
