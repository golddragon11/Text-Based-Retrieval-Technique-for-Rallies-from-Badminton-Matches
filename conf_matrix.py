# Python script to generate confusion matrix for the given data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(data, labels):
    # plt.figure(figsize=(11, 11))
    ax = sns.heatmap(data, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Stroke Type Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.show()
    # save the plot
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    data = [[ 72,   0,   1,   6,   1,   3,   0,   1,   0,   0],
            [  0,  41,   0,   0,   3,   0,   4,   0,   0,   3],
            [  2,   0,  65,  55,   4,   4,   0,  16,  10,   0],
            [ 10,   0,  23, 239,   4,   5,   1,   7,   9,   0],
            [  0,   0,   9,   5,  37,  10,  28,   8,   6,   1],
            [  4,   0,  11,   2,   3,  19,   2,  13,   5,   1],
            [  1,   6,   5,   3,  18,   1, 190,   2,   3,   3],
            [  3,   0,   5,   2,   2,   4,   0, 164,   8,   0],
            [  0,   0,   9,  13,   1,   4,   0,  28,  93,   0],
            [  0,   0,   0,   0,   2,   1,   7,   0,   5, 124]]

    labels = ['short_service', 'long_service', 'net_kill', 'net', 'push', 'drive', 'lob', 'smash', 'drop', 'clear']

    plot_confusion_matrix(data, labels)
    # calculate accuracy and precision for each class
    data = np.array(data)
    accuracy = np.diag(data) / np.sum(data, axis=1)
    precision = np.diag(data) / np.sum(data, axis=0)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)

    # calculate overall accuracy and precision
    overall_accuracy = np.sum(np.diag(data)) / np.sum(data)
    overall_precision = np.mean(precision)
    print('Overall Accuracy: ', overall_accuracy)
    print('Overall Precision: ', overall_precision)
