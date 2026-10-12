"""DIPSScore: EBMDock-derived contact and distance-likelihood scoring model."""
import torch
from torch import nn
from vendor.score_model import energy_model

class DIPSScore(nn.Module):
    def __init__(self, device='cpu', topk=1500):
        super().__init__()
        if topk < 1:
            raise ValueError('topk must be positive')
        self.topk = topk
        args = dict(device=torch.device(device), equiformer_out_dim=128,
                    hidden_dim=128, dropout_rate=0., cross_attention=False,
                    n_gaussians=6, cp=True, bsp=False, ebm_train=False,
                    dist_threshold=100., contact_threshold=15.)
        self.network = energy_model(args).to(device)

    def distributions(self, ligand, ligand_type, receptor, receptor_type):
        device = next(self.parameters()).device
        ligand, receptor = ligand.to(device), receptor.to(device)
        lb = torch.zeros(len(ligand), dtype=torch.long, device=device)
        rb = torch.zeros(len(receptor), dtype=torch.long, device=device)
        # Explicit all-pair mask: never change the set according to a candidate pose.
        mask = torch.arange(len(ligand) * len(receptor), device=device)
        params, _, contact, _ = self.network(
            ligand, lb, ligand_type.to(device), receptor, rb, receptor_type.to(device), mask)
        return params[:3], contact[0].flatten()

    def training_loss(self, ligand, ligand_type, receptor, receptor_type):
        params, logits = self.distributions(ligand, ligand_type, receptor, receptor_type)
        device = logits.device
        distances = torch.cdist(ligand.to(device), receptor.to(device)).flatten()
        labels = (distances < 15.).float()
        contact_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        selected = distances < 100.
        if not selected.any():
            raise ValueError('No native pairs within 100 Angstrom')
        nll = self.network.distance.mdn_loss_fn(
            *(p[selected] for p in params), distances[selected, None]).mean()
        return nll + contact_loss, nll.detach(), contact_loss.detach()

    def forward(self, ligand, ligand_type, receptor, receptor_type):
        params, logits = self.distributions(ligand, ligand_type, receptor, receptor_type)
        k = min(int(2.5 * (len(ligand) + len(receptor))), self.topk, logits.numel())
        indices = logits.topk(k).indices
        distances = torch.cdist(ligand.to(logits.device), receptor.to(logits.device)).flatten()
        return self.network.distance.mdn_loss_fn(
            *(p[indices] for p in params), distances[indices, None]).mean()
