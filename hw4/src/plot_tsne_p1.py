from matplotlib import pyplot as plt
import parser
import utils
from sklearn.manifold import TSNE
"""
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
"""
args = parser.arg_parse()

print('===> load features and labels ...')
train_features, train_label, valid_features, valid_labels = utils.load_features(args)

print(valid_features.shape)
valid_labels = valid_labels.squeeze(1)
print(valid_labels.shape)

# plot 11 labels
tsne = TSNE(n_components=2, random_state=0)
label_ids = range(11)
tar_features_2d = tsne.fit_transform(valid_features)

plt.figure(figsize=(6, 5))
labels_colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'brown'
label_names = [0,1,2,3,4,5,6,7,8,9,10]
for i, c, label in zip(label_ids, labels_colors, label_names):
    plt.scatter(tar_features_2d[valid_labels == i, 0], tar_features_2d[valid_labels == i, 1], c=c, label=label, s=1)
plt.legend()
plt.savefig('tsne_p1.png')



"""
digits = datasets.load_digits()
print(digits.target_names)
#print(digits.data.shape) (1797, 64)
#print(digits.target.shape) (1797,)
#print(digits.target_names.shape) (10,)
#print(digits.images.shape) (1797, 8, 8)
#data, target, target_names, images
"""
