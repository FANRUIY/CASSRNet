import torch
import torch.nn as nn
import torch.nn.functional as F


class class_loss_3class(nn.Module):
    # Class loss
    def __init__(self):
        super(class_loss_3class, self).__init__()

    def forward(self, type_res):
        n = len(type_res)
        m = len(type_res[0]) - 1
        type_all = type_res
        loss = 0
        for i in range(n):
            sum_re = abs(type_all[i][0] - type_all[i][1]) + abs(type_all[i][0] - type_all[i][2]) + abs(type_all[i][1] - type_all[i][2])
            loss += (m - sum_re)
        return loss / n


class average_loss_3class(nn.Module):
    # Average loss
    def __init__(self):
        super(average_loss_3class, self).__init__()

    def forward(self, type_res):
        n = len(type_res)
        m = len(type_res[0])
        type_all = type_res
        sum1 = 0
        sum2 = 0
        sum3 = 0

        for i in range(n):
            sum1 += type_all[i][0]
            sum2 += type_all[i][1]
            sum3 += type_all[i][2]

        return (abs(sum1 - n / m) + abs(sum2 - n / m) + abs(sum3 - n / m)) / ((n / m) * 4)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class GANLoss(nn.Module):
    """Define GAN loss: [vanilla | lsgan | wgan-gp]"""
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(
            outputs=interp_crit, inputs=interp,
            grad_outputs=grad_outputs, create_graph=True,
            retain_graph=True, only_inputs=True
        )[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)
        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


# ---------- 新增强力损失组合 ----------

def ssim_loss(x, y, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    eps = 1e-4  # 防止除零，适当调大

    # 输入先裁剪到0~1范围，避免异常数值
    x = torch.clamp(x, 0.0, 1.0)
    y = torch.clamp(y, 0.0, 1.0)

    mu_x = F.avg_pool2d(x, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, 1, window_size // 2)

    sigma_x = F.avg_pool2d(x * x, window_size, 1, window_size // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, 1, window_size // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, window_size // 2) - mu_x * mu_y

    sigma_x = torch.clamp(sigma_x, min=0.0)
    sigma_y = torch.clamp(sigma_y, min=0.0)

    denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    denominator = torch.clamp(denominator, min=eps)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / denominator

    ssim = (1 - ssim_map.mean()) / 2
    if torch.isnan(ssim) or torch.isinf(ssim):
        print('[SSIM] 检测到 NaN 或 Inf，自动重置为0')
        ssim = torch.tensor(0.0, device=x.device)

    return torch.clamp(ssim, 0.0, 1.0)




def edge_loss(pred, target):
    """
    Edge loss using Sobel operator, computes L1 diff of edges.
    """
    C = pred.shape[1]

    sobel_x = torch.tensor([[[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]]], dtype=pred.dtype, device=pred.device).repeat(C, 1, 1, 1)
    sobel_y = torch.tensor([[[[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]]]], dtype=pred.dtype, device=pred.device).repeat(C, 1, 1, 1)

    def get_edges(img):
        edges_x = F.conv2d(img, sobel_x, padding=1, groups=C)
        edges_y = F.conv2d(img, sobel_y, padding=1, groups=C)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-6)
        return edges.mean(dim=1, keepdim=True)

    pred_edges = get_edges(pred)
    target_edges = get_edges(target)

    return F.l1_loss(pred_edges, target_edges)


class StrongLoss(nn.Module):
    """
    Combined loss: Charbonnier + SSIM + Edge
    """
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
        super(StrongLoss, self).__init__()
        self.l1 = CharbonnierLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_l = ssim_loss(pred, target)
        edge_l = edge_loss(pred, target)
        loss = self.alpha * l1_loss + self.beta * ssim_l + self.gamma * edge_l
        return loss
