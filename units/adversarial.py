import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import pywt
def fgsm_attack(model, x, epsilon=0.1):
    x.requires_grad = True
    model.train()
    _, _, _, _, loss = model(x)
    model.zero_grad()
    loss.backward()
    sign_grad = x.grad.data.sign()
    x_adv = torch.clamp(x + epsilon * sign_grad, 0, 1)
    return x_adv
def pgd_attack(model, x, epsilon=0.1, alpha=0.01, num_iter=10):
    x_adv = x.clone().detach().requires_grad_(True)
    # with torch.enable_grad():
    for _ in range(num_iter):
        _, _, _, _, loss = model(x_adv)
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.zero_()
        loss.backward()

        sign_grad = x_adv.grad.data.sign()
        x_adv = x_adv + alpha * sign_grad
        eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon) 
        x_adv = torch.clamp(x + eta, min = 0, max = 1).detach()
        x_adv.requires_grad = True
    return x_adv

def cw_attack(model, x, lr = 0.01, num_iter = 10, c=1):
    batch, trans = x.shape[0], x.shape[1]
    delta = torch.zeros_like(x, requires_grad = True, device = x.device)
    optimizer_attack = optim.Adam([delta], lr=lr)
    kappa = 0
    for i in range(num_iter):
        x_adv = torch.clamp(x + delta, min = 0, max = 1)
        # _, _, _, _, total_loss = model(x_adv)
        g = model.selfsupervision(model.encoder(x_adv)) # (batch ,trans, trans)
        one_hot_labels = F.one_hot(torch.tensor([i for i in range(trans) for _ in range(batch)], device=x.device), num_classes=trans).float()

        correct_logit = torch.sum(one_hot_labels * F.log_softmax(g.reshape(-1, trans), dim=1), dim=1)
        wrong_logit, _ = torch.max((1 - one_hot_labels) * F.log_softmax(g.reshape(-1, trans), dim=1) - one_hot_labels * 1e4, dim=1)
        # f(x) = max(correct_logit - wrong_logit + kappa, 0)
        f_loss = torch.clamp(correct_logit - wrong_logit + kappa, min=0)
        l2_loss = torch.sum(delta.reshape(batch * trans, x.shape[2], x.shape[3], x.shape[4]) ** 2, dim=[1, 2, 3])
        loss = torch.sum(l2_loss + c * f_loss)
        optimizer_attack.zero_grad()
        loss.backward()
        optimizer_attack.step()
    x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()
def dwt(x, wavelet='db4', level=3):
    device = x.device
    coeffs = pywt.wavedec(x.detach().cpu().numpy(), wavelet = wavelet, level=level)
    coeffs_torch = [torch.tensor(c, device=device, requires_grad=True) for c in coeffs]
    return coeffs_torch
def idwt(coeffs, wavelet='db4', original_length=None):
    device = coeffs[0].device
    coeffs_np = [c.detach().cpu().numpy() for c in coeffs]
    x_rec = pywt.waverec(coeffs_np, wavelet)
    if original_length is not None:
        x_rec = x_rec[..., :original_length]
    return torch.tensor(x_rec, device=device, requires_grad=True)

def tf_attack(model, x, lr=0.01, num_iter = 10):
    batch, trans, _, n_times, chs = x.shape
    x.requires_grad = True
    model.train()
    x_adv = x.clone().detach().to(x.device).requires_grad_(True)
    for _ in range(num_iter):
        if x_adv.grad is not None:
            x_adv.grad.data.zero_()
        _, _, _, _, total_time_loss = model(x_adv)
        model.zero_grad()
        total_time_loss.backward()
        grad = x_adv.grad.data.sign()
        x_adv = x_adv + lr * grad
        x_adv = torch.clamp(x_adv, 0, 1).detach().requires_grad_(True)

        _, _, _, _, total_freq_loss = model(x_adv)
        model.zero_grad()
        total_freq_loss.backward()
        grad = x_adv.grad.data.sign()
        grad = grad.permute(0, 1, 2, 4, 3)
        x_reshape = x_adv.permute(0, 1, 2, 4, 3)
        x_reshape = x_reshape.reshape(batch * trans * 1 * chs, n_times)
        coeffs_torch = dwt(x_reshape)
        for i in range(len(coeffs_torch)):
            coeffs_torch[i] = coeffs_torch[i] + lr * torch.mean(grad.reshape(batch * trans * 1 * chs, n_times), dim=-1, keepdim=True)
        x_adv = idwt(coeffs_torch, original_length=x_reshape.shape[-1]).reshape(batch, trans, 1, chs, n_times)
        x_adv = x_adv.permute(0, 1, 2, 4, 3)
        x_adv = torch.clamp(x_adv, 0, 1).detach().requires_grad_(True)
    return x_adv.detach()

if __name__ == '__main__':
    pass



