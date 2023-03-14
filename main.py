import numpy as np
import torch
import torchvision
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from pyod.models.knn import KNN

from dataset import NormalCIFAR10Dataset, AnomalousCIFAR10Dataset, NormalCIFAR10DatasetRotationAugmented, AnomalousCIFAR10DatasetRotationAugmented
from transforms import Transform
from model import Model, AlexNet, ResNetModel
from common import norm_of_kde
from common import get_indices_with_lowest



BATCH_SIZE = 128
PROJECTION_DIM = 256
EPOCHS = 200
LEARNING_RATE = 3e-6


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)


def get_description(loss, mean_loss, kde_loss):
    return f'Loss: {loss:.4f}, Mean Loss: {mean_loss:.4f}, KDE Loss: {kde_loss:.4f}'


def get_cosine_sim_matrix(X):
    print(X.shape)
    a = X @ X.transpose(0, 1)
    b = X.norm(dim=1).reshape(-1, 1) @ X.norm(dim=1).reshape(1, -1)
    return (a) / (b)


def save_anomaly_score_samples(anomaly_scores, test_data_normal, test_data_anomalous):
    lowest_normal = get_indices_with_lowest(anomaly_scores[0:len(anomaly_scores) // 2], 10)
    for i in range(len(lowest_normal)):
        save_image(test_data_normal[lowest_normal[i]][0], f'./results/lowest_normal_{i}.png')

    highest_normal = get_indices_with_lowest([-x for x in anomaly_scores[0:len(anomaly_scores) // 2]], 10)
    for i in range(len(highest_normal)):
        save_image(test_data_normal[highest_normal[i]][0], f'./results/highest_normal_{i}.png')

    lowest_anomalous = get_indices_with_lowest(anomaly_scores[len(anomaly_scores) // 2:], 10)
    for i in range(len(lowest_anomalous)):
        save_image(test_data_anomalous[lowest_anomalous[i]][0], f'./results/lowest_anomalous_{i}.png')

    highest_anomalous = get_indices_with_lowest([-x for x in anomaly_scores[len(anomaly_scores) // 2:]], 10)
    for i in range(len(highest_anomalous)):
        save_image(test_data_anomalous[highest_anomalous[i]][0], f'./results/highest_anomalous_{i}.png')


def evaluate_auroc_knn(models, projection_size, train_loader, test_loader_normal, test_loader_anomalous):
    X_train = np.zeros((0, 512))

    for xs in train_loader:
        z_sum = torch.zeros((xs[0].shape[0], projection_size)).to(device)
        for model, x in zip(models, xs):
            x = x.to(device)
            z = model(x)
            z_sum += z
        X_train = np.concatenate((X_train, z_sum.cpu().detach().numpy()), axis=0)

    knn = KNN(n_neighbors=1)
    knn.fit(X_train)

    X_test = np.zeros((0, projection_size))
    y_test = np.zeros(0)

    for loader, type in [(test_loader_normal, 'normal'), (test_loader_anomalous, 'anomalous')]:
        for xs in loader:
            z_sum = torch.zeros(1, projection_size).to(device)
            for model, x in zip(models, xs):
                x = x.to(device)
                z = model(x)
                z_sum += z
            X_test = np.concatenate((X_test, z_sum.cpu().detach().numpy()), axis=0)
            y_test = np.concatenate((y_test, np.array([0 if type == 'normal' else 1])), axis=0)

    anomaly_scores = knn.decision_function(X_test)

    save_anomaly_score_samples(anomaly_scores, test_data_normal, test_data_anomalous)

    return roc_auc_score(y_test, anomaly_scores)


def evaluate_auroc(models, projection_size, test_loader_normal, test_loader_anomalous, save_sample_figs=False, test_data_normal=None, test_data_anomalous=None):
    assert not save_sample_figs or (test_data_normal is not None and test_data_anomalous is not None)

    for model in models:
        model.eval()

    y_test = []
    anomaly_scores = []

    for loader, type in [(test_loader_normal, 'normal'), (test_loader_anomalous, 'anomalous')]:
        for xs in loader:
            anomaly_score = 0
            for i in range(4):
                z_sum = torch.zeros(1, projection_size).to(device)
                for model, x in zip(models[3*i:3*(i+1)], [xs[i] for j in range(3)]):
                    x = x.to(device)
                    z = model(x)
                    z_sum += z
                anomaly_score += (z_sum ** 2).sum(dim=1)[0].item()
            anomaly_scores.append(anomaly_score)
            y_test.append(0 if type == 'normal' else 1)

    save_anomaly_score_samples(anomaly_scores, test_data_normal, test_data_anomalous)

    return roc_auc_score(y_test, anomaly_scores)


def visualize_tsne(z_all, batch_size):
    z_tsne_embedded = TSNE(n_components=2).fit_transform(z_all)
    figure = plt.figure()
    plt.scatter(
        z_tsne_embedded[:, 0],
        z_tsne_embedded[:, 1],
        c=['red' for i in range(batch_size)] + ['blue' for i in range(batch_size)]
          # + ['green' for i in range(batch_size)] + ['yellow' for i in range(batch_size)]
          # + ['orange' for i in range(batch_size)]
        # + ['orange' for i in range(batch_size)] + ['purple' for i in range(batch_size)] + ['pink' for i in range(batch_size)] + ['black' for i in range(batch_size)]
        ,
        marker='2'
    )
    plt.savefig('tsne.png')


for normal_class in range(0, 10):
    train_data = NormalCIFAR10Dataset(normal_class, train=True, transform=Transform(test=False))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    test_data_normal = NormalCIFAR10Dataset(normal_class, train=False, transform=Transform(test=True))
    test_loader_normal = torch.utils.data.DataLoader(test_data_normal, batch_size=1, shuffle=False, drop_last=False)

    test_data_anomalous = torch.utils.data.Subset(AnomalousCIFAR10Dataset(normal_class, train=False, transform=Transform(test=True)), list(range(len(test_data_normal))))
    test_loader_anomalous = torch.utils.data.DataLoader(test_data_anomalous, batch_size=1, shuffle=False, drop_last=False)

    models = []
    optimizers_models = []

    for i in range(3 * 4):
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

        if epoch % 5 == 0:
            print(f'AUROC: {(100 * evaluate_auroc(models, 256, test_loader_normal, test_loader_anomalous, save_sample_figs=True, test_data_normal=test_data_normal, test_data_anomalous=test_data_anomalous)):.4f}%')
            # print(f'AUROC: {(100 * evaluate_auroc([model.backbone for model in models], 512, test_loader_normal, test_loader_anomalous, save_sample_figs=True, test_data_normal=test_data_normal, test_data_anomalous=test_data_anomalous)):.4f}%')
            # print(f'AUROC: {(100 * evaluate_auroc_knn([model.backbone for model in models], 512, train_loader, test_loader_normal, test_loader_anomalous)):.4f}%')

        for model in models:
            model.train()

        for batch, xs in enumerate(iterator):
            mean_loss = torch.tensor(0.0, device=device)
            kde_loss = torch.tensor(0.0, device=device)

            for i in range(4):
                z_sum = torch.zeros(BATCH_SIZE, PROJECTION_DIM).to(device)
                z_all = torch.zeros(0, PROJECTION_DIM).to(device)
                zs = []

                for model, x in zip(models[3*i:3*(i+1)], [xs[i] for j in range(3)]):
                    model.zero_grad()
                    x = x.to(device)
                    z = model(x)
                    zs.append(z)
                    z_sum += z
                    z_all = torch.cat((z_all, z), dim=0)

                mean_loss += (z_sum ** 2).sum(dim=1).mean()

                # if batch == 0:
                #     visualize_tsne(z_all.detach().cpu().numpy(), BATCH_SIZE)


                kde_loss += norm_of_kde(z_all, 1)

            # kde_loss = torch.as_tensor(0.0, device=device)
            #
                # for z in zs:
                #     kde_loss += norm_of_kde(z, 0.5)

            loss = 0.3 * mean_loss + 0.3 * kde_loss

            summed_mean_loss += mean_loss.item()
            summed_kde_loss += kde_loss.item()
            summed_loss += loss.item()
            iterator.set_description(get_description(summed_loss / (batch + 1), summed_mean_loss / (batch + 1), summed_kde_loss / (batch + 1)))

            loss.backward()
            for optimizer_model in optimizers_models:
                optimizer_model.step()

        # print(get_cosine_sim_matrix(torch.stack((mean_0, mean_1, mean_2, mean_3), dim=1).transpose(0, 1)))


