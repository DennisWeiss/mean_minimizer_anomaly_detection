import torch
import torchvision
from tqdm import tqdm

from dataset import NormalCIFAR10Dataset, AnomalousCIFAR10Dataset, NormalCIFAR10DatasetRotationAugmented, AnomalousCIFAR10DatasetRotationAugmented
from transforms import Transform
from model import Model
from common import norm_of_kde
from sklearn.metrics import roc_auc_score


EPOCHS = 100
LEARNING_RATE = 3e-6


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_description(loss, mean_loss, kde_loss):
    return f'Loss: {loss:.4f}, Mean Loss: {mean_loss:.4f}, KDE Loss: {kde_loss:.4f}'


def get_cosine_sim_matrix(X):
    print(X.shape)
    a = X @ X.transpose(0, 1)
    b = X.norm(dim=1).reshape(-1, 1) @ X.norm(dim=1).reshape(1, -1)
    return (a) / (b)


def evaluate_auroc(models, test_loader_normal, test_loader_anomalous):
    for model in models:
        model.eval()

    y_test = []
    anomaly_scores = []

    for loader, type in [(test_loader_normal, 'normal'), (test_loader_anomalous, 'anomalous')]:
        for [x0, x1, x2, x3] in loader:
            x0, x1, x2, x3 = x0.to(device), x1.to(device), x2.to(device), x3.to(device)
            z0, z1, z2, z3 = models[0](x0), models[1](x1), models[2](x2), models[3](x3)
            anomaly_scores.append(((z0 + z1 + z2 + z3) ** 2).sum(dim=1)[0].item())
            y_test.append(0 if type == 'normal' else 1)

    return roc_auc_score(y_test, anomaly_scores)


for normal_class in range(1, 2):
    train_data = NormalCIFAR10Dataset(normal_class, train=True, transform=Transform())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, drop_last=True)

    test_data_normal = NormalCIFAR10Dataset(normal_class, train=False, transform=Transform())
    test_loader_normal = torch.utils.data.DataLoader(test_data_normal, batch_size=1, shuffle=False, drop_last=False)

    test_data_anomalous = torch.utils.data.Subset(AnomalousCIFAR10Dataset(normal_class, train=False, transform=Transform()), list(range(len(test_data_normal))))
    test_loader_anomalous = torch.utils.data.DataLoader(test_data_anomalous, batch_size=1, shuffle=False, drop_last=False)

    models = []
    optimizers_models = []
    for i in range(4):
        model = Model().to(device)
        optimizer_model = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        models.append(model)
        optimizers_models.append(optimizer_model)

    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}')
        iterator = tqdm(train_loader)

        summed_mean_loss = 0
        summed_kde_loss = 0
        summed_loss = 0

        mean_0 = torch.zeros(256).to(device)
        mean_1 = torch.zeros(256).to(device)
        mean_2 = torch.zeros(256).to(device)
        mean_3 = torch.zeros(256).to(device)

        if epoch < 10 or epoch % 5 == 0:
            print(f'AUROC: {(100 * evaluate_auroc(models, test_loader_normal, test_loader_anomalous)):.4f}%')

        for model in models:
            model.train()

        for batch, [x0, x1, x2, x3] in enumerate(iterator):
            x0, x1, x2, x3 = x0.to(device), x1.to(device), x2.to(device), x3.to(device)

            for optimizer_model in optimizers_models:
                optimizer_model.zero_grad()

            z0, z1, z2, z3 = models[0](x0), models[1](x1), models[2](x2), models[3](x3)

            mean_0 = mean_0 + (z0.mean(dim=0) - mean_0) / (batch + 1)
            mean_1 = mean_1 + (z1.mean(dim=0) - mean_1) / (batch + 1)
            mean_2 = mean_2 + (z2.mean(dim=0) - mean_2) / (batch + 1)
            mean_3 = mean_3 + (z3.mean(dim=0) - mean_3) / (batch + 1)

            mean_loss = ((z0 + z1 + z2 + z3) ** 2).sum(dim=1).mean()
            # z_all = torch.cat((z0, z1, z2, z3), dim=0)

            # kde_loss = norm_of_kde(z_all, 0.1)

            kde_loss = torch.as_tensor(0.0, device=device)

            for z in [z0, z1, z2, z3]:
                kde_loss += norm_of_kde(z, 0.5)

            loss = mean_loss + 1 * kde_loss

            summed_mean_loss += mean_loss.item()
            summed_kde_loss += kde_loss.item()
            summed_loss += loss.item()
            iterator.set_description(get_description(summed_loss / (batch + 1), summed_mean_loss / (batch + 1), summed_kde_loss / (batch + 1)))

            loss.backward()
            for optimizer_model in optimizers_models:
                optimizer_model.step()

        # print(get_cosine_sim_matrix(torch.stack((mean_0, mean_1, mean_2, mean_3), dim=1).transpose(0, 1)))


