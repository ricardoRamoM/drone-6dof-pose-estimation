# model.py
# Regresor de pose 6-DoF basado en ResNet-18 preentrenada en ImageNet.
# La capa fully-connected final (originalmente 1000 clases) se reemplaza
# por una capa lineal de 7 salidas:
#   - 3 para posición (x, y, z)
#   - 4 para orientación (cuaternión qx, qy, qz, qw)
#
# Se normalizan los cuaterniones de salida para garantizar rotaciones válidas.
# La pérdida combina MSE de posición + (1 - |dot|) entre cuaterniones
# unitarios, ya que q y -q representan la misma rotación.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ResNetPoseRegressor(nn.Module):
    """ResNet-18 para regresión de posición y orientación 6-DoF."""

    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            7,
        )

    def forward(self, images):
        return self.backbone(images)


def normalize_quaternion(quaternion, eps=1e-8):
    """Normaliza cuaterniones con orden [qx, qy, qz, qw]."""
    return F.normalize(quaternion, p=2, dim=-1, eps=eps)


def pose_loss(
    prediction,
    target,
    position_mean=None,
    position_std=None,
    position_weight=1.0,
    orientation_weight=1.0,
):
    """
    Pérdida conjunta de posición y orientación.

    Parámetros
    ----------
    prediction : Tensor (B, 7)
        Salida del modelo: [x, y, z, qx, qy, qz, qw].
    target : Tensor (B, 7)
        Pose objetivo. Si ``position_mean`` / ``position_std`` se
        proporcionan, se asume que ``target[:, :3]`` está en unidades
        originales y se normaliza dentro de la función.
    position_mean, position_std : Tensor (3,) o None
        Si ambos se pasan, el target de posición se normaliza como
        ``(target[:, :3] - mean) / std`` antes de la MSE.
        La predicción se asume ya normalizada por el modelo.
    position_weight, orientation_weight : float
        Pesos de cada término en la pérdida total.

    Devuelve
    --------
    (total_loss, position_loss.detach(), orientation_loss.detach())
    """
    predicted_position = prediction[:, :3]
    predicted_quaternion = normalize_quaternion(prediction[:, 3:7])

    target_position = target[:, :3]
    if position_mean is not None and position_std is not None:
        target_position = (target_position - position_mean) / position_std

    target_quaternion = normalize_quaternion(target[:, 3:7])

    position_loss = F.mse_loss(predicted_position, target_position)

    quaternion_dot = torch.sum(
        predicted_quaternion * target_quaternion, dim=1
    ).abs().clamp(0.0, 1.0)

    orientation_loss = (1.0 - quaternion_dot).mean()

    total_loss = (
        position_weight * position_loss
        + orientation_weight * orientation_loss
    )

    return total_loss, position_loss.detach(), orientation_loss.detach()