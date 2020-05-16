import models
from matplotlib import pyplot as plt
import data
import parser
import torch
#torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from sklearn.manifold import TSNE


args = parser.arg_parse()
tar_test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset=args.target_dataset),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)
src_test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset=args.source_dataset),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)

feature_extractor, label_predictor, domain_classifier = models.DANN(args)
feature_extractor.cuda(), label_predictor.cuda()
feature_extractor.load_state_dict(torch.load(args.resume_folder + 'feature_extractor.pth.tar'))
label_predictor.load_state_dict(torch.load(args.resume_folder + 'label_predictor.pth.tar'))
feature_extractor.eval(), label_predictor.eval()


tar_features, tar_labels, src_features, src_labels = [], [], [], []
with torch.no_grad(): # do not need to calculate information for gradient during eval
    for idx, (imgs, classes) in enumerate(tar_test_loader):
        imgs = imgs.cuda()
        features = feature_extractor(imgs).view(-1, 512).detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()
        tar_features.append(features)
        tar_labels.append(classes)
    for idx, (imgs, classes) in enumerate(src_test_loader):
        imgs = imgs.cuda()
        features = feature_extractor(imgs).view(-1, 512).detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()
        src_features.append(features)
        src_labels.append(classes)

tar_features = np.concatenate(tar_features)
tar_labels = np.concatenate(tar_labels)
tar_domain = np.ones(tar_labels.shape[0])

src_features = np.concatenate(src_features)
src_labels = np.concatenate(src_labels)
src_domain = np.zeros(src_labels.shape[0])

total_features = np.concatenate((tar_features, src_features))
total_labels = np.concatenate((tar_labels, src_labels))
total_domains = np.concatenate((tar_domain, src_domain))
print(total_features.shape)
print(total_labels.shape)
print(total_domains.shape)


# plot 10 labels
tsne = TSNE(n_components=2, random_state=0)
label_ids = range(10)
tar_features_2d = tsne.fit_transform(total_features)

plt.figure(figsize=(6, 5))
labels_colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
label_names = [0,1,2,3,4,5,6,7,8,9]
for i, c, label in zip(label_ids, labels_colors, label_names):
    plt.scatter(tar_features_2d[total_labels == i, 0], tar_features_2d[total_labels == i, 1], c=c, label=label, s=0.01)
plt.legend()
plt.savefig('tsne_label.png')



# plot 2 domains
plt.figure(figsize=(6, 5))
domain_ids = range(2)
domain_colors = 'b', 'g'
domain_names = [0,1]
for i, c, label in zip(domain_ids, domain_colors, domain_names):
    plt.scatter(tar_features_2d[total_domains == i, 0], tar_features_2d[total_domains == i, 1], c=c, label=label, s=0.01)
plt.legend()
plt.savefig('tsne_domain.png')



"""
digits = datasets.load_digits()
print(digits.target_names)
#print(digits.data.shape) (1797, 64)
#print(digits.target.shape) (1797,)
#print(digits.target_names.shape) (10,)
#print(digits.images.shape) (1797, 8, 8)
#data, target, target_names, images
"""
