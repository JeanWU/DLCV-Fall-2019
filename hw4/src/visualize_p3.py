import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#pred = 'p3_log/results/OP04-R04-ContinentalBreakfast.txt'
#gt = 'p3_log/results/OP04-R04-ContinentalBreakfast_gt.txt'
pred = '../results/OP04-R04-ContinentalBreakfast.txt'
gt = '../results/OP04-R04-ContinentalBreakfast_gt.txt'

preds, gts = [], []

with open(pred, 'r') as f:
    for line in f:
        preds.append(int(line.strip('\n')))

with open(gt, 'r') as f:
    for line in f:
        gts.append(int(line.strip('\n')))

plt.figure(figsize=(16,4))
ax = plt.subplot(211)

#colors = ["wheat", "turqoise", "teal", "sienna", "salmon", "orange",
#           "lightblue", "lavender", "gold", "darkblue", "azure"]
colors = plt.cm.get_cmap('tab20',11).colors
plt.scatter(np.arange(11),np.ones(11), c=colors, s=180)
plt.savefig("color_num.png")
#colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'brown'
cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in preds])
# cmap = plt.cm.get_cmap("tab20", 11)
bounds = [i for i in range(len(preds))]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                       norm=norm,
                                       boundaries=bounds,
                                       spacing='proportional',
                                       orientation='horizontal')
ax.set_ylabel('Prediction')

ax2 = plt.subplot(212)
cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in gts])
bounds = [i for i in range(len(gts))]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
cb2 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
                                       norm=norm,
                                       boundaries=bounds,
                                       spacing='proportional',
                                       orientation='horizontal')


ax2.set_ylabel('GroundTruth')

plt.savefig("temporal_action_segmentation_v5.png")
