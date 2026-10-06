import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# 1. Configuración del dispositivo
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("Dispositivo:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. Crear ResNet-18
# ============================================================

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Cambiar la capa final:
# 1000 clases -> 6 valores
model.fc = nn.Linear(model.fc.in_features, 6)

# Mover modelo al dispositivo
model = model.to(device)

print("\nModelo:")
print(model.fc)


# ============================================================
# 3. Crear una imagen ficticia
# ============================================================

# 1 imagen
# 3 canales RGB
# 224 x 224 píxeles
x = torch.randn(1, 3, 224, 224).to(device)

# Pose ficticia de referencia
y = torch.randn(1, 6).to(device)


# ============================================================
# 4. Forward pass
# ============================================================

prediction = model(x)

print("\nEntrada:", x.shape)
print("Salida:", prediction.shape)
print("Predicción:", prediction)


# ============================================================
# 5. Loss + Backpropagation
# ============================================================

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Guardar pesos antes
weight_before = model.fc.weight.detach().clone()

# Forward
prediction = model(x)

# Loss
loss = criterion(prediction, y)

# Backpropagation
optimizer.zero_grad()
loss.backward()

# Actualizar pesos
optimizer.step()

# Guardar pesos después
weight_after = model.fc.weight.detach().clone()


# ============================================================
# 6. Resultados
# ============================================================

print("\n--- Entrenamiento ---")
print("Loss:", loss.item())
print("Gradiente calculado:", model.fc.weight.grad is not None)
print("Pesos actualizados:", not torch.equal(weight_before, weight_after))