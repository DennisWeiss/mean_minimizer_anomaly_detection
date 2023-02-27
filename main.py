import torch
import torchvision
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dataset import NormalCIFAR10Dataset, AnomalousCIFAR10Dataset, NormalCIFAR10DatasetRotationAugmented, AnomalousCIFAR10DatasetRotationAugmented
from transforms import Transform
from model import Model
from common import norm_of_kde



BATCH_SIZE = 256
PROJECTION_DIM = 256
EPOCHS = 200
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
        for xs in loader:
            z_sum = torch.zeros(BATCH_SIZE, PROJECTION_DIM).to(device)
            for model, x in zip(models, xs):
                x = x.to(device)
                z = model(x)
                z_sum += z
            anomaly_scores.append((z_sum ** 2).sum(dim=1)[0].item())
            y_test.append(0 if type == 'normal' else 1)

    return roc_auc_score(y_test, anomaly_scores)


def visualize_tsne(z_all, batch_size):
    z_tsne_embedded = TSNE(n_components=2).fit_transform(z_all)
    figure = plt.figure()
    plt.scatter(
        z_tsne_embedded[:, 0],
        z_tsne_embedded[:, 1],
        c=['red' for i in range(batch_size)] + ['blue' for i in range(batch_size)] + ['green' for i in range(batch_size)] + ['yellow' for i in range(batch_size)],
        marker='2'
    )
    plt.savefig('tsne.png')


for normal_class in range(3, 4):
    train_data = NormalCIFAR10Dataset(normal_class, train=True, transform=Transform())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

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

        if epoch < 10 or epoch % 5 == 0:
            print(f'AUROC: {(100 * evaluate_auroc(models, test_loader_normal, test_loader_anomalous)):.4f}%')

        for model in models:
            model.train()

        for batch, xs in enumerate(iterator):
            z_sum = torch.zeros(BATCH_SIZE, PROJECTION_DIM).to(device)
            z_all = torch.zeros(0, PROJECTION_DIM).to(device)
            zs = []

            for model, x in zip(models, xs):
                model.zero_grad()
                x = x.to(device)
                z = model(x)
                zs.append(z)
                z_sum += z
                z_all = torch.cat((z_all, z), dim=0)

            mean_loss = (z_sum ** 2).sum(dim=1).mean()

            if batch == 0:
                visualize_tsne(z_all.detach().cpu().numpy(), BATCH_SIZE)

            kde_loss = norm_of_kde(z_all, 0.5)

            # kde_loss = torch.as_tensor(0.0, device=device)
            #
            # for z in zs:
            #     kde_loss += norm_of_kde(z, 0.2)

            loss = mean_loss + 1 * kde_loss

            summed_mean_loss += mean_loss.item()
            summed_kde_loss += kde_loss.item()
            summed_loss += loss.item()
            iterator.set_description(get_description(summed_loss / (batch + 1), summed_mean_loss / (batch + 1), summed_kde_loss / (batch + 1)))

            loss.backward()
            for optimizer_model in optimizers_models:
                optimizer_model.step()

        # print(get_cosine_sim_matrix(torch.stack((mean_0, mean_1, mean_2, mean_3), dim=1).transpose(0, 1)))


