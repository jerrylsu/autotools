import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
from itertools import cycle
from sklearn.metrics import roc_curve, auc


def Visualization():
    fig_1 = plt.subplot(121)
    fig_2 = plt.subplot(122)
    pass


def plot_3d(encode, labels):
    fig = plt.figure(num="SDAE")  # 画布
    ax = Axes3D(fig)
    X, Y, Z = encode[:, 0], encode[:, 1], encode[:, 2]
    for x, y, z, s in zip(X, Y, Z, labels):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x=x, y=y, z=z, s=s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()


def plot_2d(predicts, labels):
    fig = plt.figure(num="SDAE")  # 画布
    X, Y = predicts[:, 0], predicts[:, 1]
    C = ['r' if label == 0 else 'b' for label in labels]
    for x, y, c in zip(X, Y, C):
        plt.scatter(x=x, y=y, s=75, c=c, alpha=0.5)
    plt.xlim(X.min(), X.max())
    plt.ylim((Y.min(), Y.max()))


def plot_roc(predicts, labels, name: str):
    """Compute and plot ROC & AUC.
    predicts: [batch_size, n_classes]
    labels: [batch_size,]
    name: string
    """
    # Compute ROC curve and ROC area for each class
    labels = torch.nn.functional.one_hot(torch.from_numpy(labels), num_classes=2).numpy()
    n_classes = predicts.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predicts[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predicts.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'micro-average ROC curve (area = {round(roc_auc["micro"], 2)})')
    colors = cycle(['aqua', 'darkorange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=2, label=f'ROC curve of class {i} (area = {round(roc_auc[i], 2)})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    plt.show()

