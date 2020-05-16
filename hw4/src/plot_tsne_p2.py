from matplotlib import pyplot as plt
import parser
import utils
from sklearn.manifold import TSNE
import models
import torch
import numpy as np

args = parser.arg_parse()

print('===> load features and labels ...')
train_features, train_label, valid_features, valid_labels = utils.load_features(args)

''' load model '''
print('===> prepare model ...')
feature_extractor, BiRNN = models.P2(args)
BiRNN.load_state_dict(torch.load('p2_log/BiRNN_best_0.495.pth.tar'))

BiRNN.cuda()
BiRNN.eval()

hidden_features = []
with torch.no_grad():
    total_length = valid_features.shape[0]
    print("total_length: ",total_length)
    """ using batch size = 1, so no need to worry about padding and sequence length """
    for index in range(0,total_length):
        input_X = torch.tensor(valid_features[index])
        input_X, input_y, lengths = utils.single_batch_padding([input_X],[valid_labels[index]],test=True)

        input_X = input_X.cuda()
        _, hidden = BiRNN(input_X, lengths)
        hidden = hidden.detach().cpu().numpy()
        hidden_features.append(hidden)

# hidden_features.shape [(2653, 1024)]
hidden_features = np.array(hidden_features)
hidden_features = np.concatenate(hidden_features)


print(hidden_features.shape)
#print(train_features.shape)
valid_labels = valid_labels.squeeze(1)
print(valid_labels.shape)

# plot 11 labels
tsne = TSNE(n_components=2, random_state=0)
label_ids = range(11)
tar_features_2d = tsne.fit_transform(hidden_features)

plt.figure(figsize=(6, 5))
labels_colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'brown'
label_names = [0,1,2,3,4,5,6,7,8,9,10]
for i, c, label in zip(label_ids, labels_colors, label_names):
    plt.scatter(tar_features_2d[valid_labels == i, 0], tar_features_2d[valid_labels == i, 1], c=c, label=label, s=1)
plt.legend()
plt.savefig('tsne_p2.png')



"""
digits = datasets.load_digits()
print(digits.target_names)
#print(digits.data.shape) (1797, 64)
#print(digits.target.shape) (1797,)
#print(digits.target_names.shape) (10,)
#print(digits.images.shape) (1797, 8, 8)
#data, target, target_names, images
"""
