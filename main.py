import torch
import torchvision
from tqdm import tqdm

from dataset import NormalCIFAR10Dataset, AnomalousCIFAR10Dataset, NormalCIFAR10DatasetRotationAugmented, AnomalousCIFAR10DatasetRotationAugmented
from transforms import Transform
from model import Model
from common import norm_of_kde


EPOCHS = 100
LEARNING_RATE = 1e-4


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

for normal_class in range(0, 10):
    train_data = NormalCIFAR10Dataset(normal_class, train=True, transform=Transform())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    model = Model().to(device)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    iterator = tqdm(train_loader)

    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}')
        for x0, x1 in iterator:
            x0, x1 = x0.to(device), x1.to(device)

            optimizer_model.zero_grad()

            z0, z1 = model(x0), model(x1)

            mean_loss = ((z0 + z1) ** 2).sum(dim=1).mean()
            z_all = torch.cat((z0, z1), dim=0)
            kde_loss = norm_of_kde(z_all)
            loss = mean_loss + kde_loss

            loss.backward()
            optimizer_model.step()



