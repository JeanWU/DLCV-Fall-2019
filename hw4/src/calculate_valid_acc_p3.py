import sys
import os
dir = sys.argv[1]
# path to all result file

full_video_path = '../hw4_data/FullLengthVideos/videos/valid/'
category_list = sorted(os.listdir(os.path.join(full_video_path)))

all_acc = []
for category in category_list:
    pred = os.path.join(dir, '{}.txt'.format(category))
    gt = os.path.join(dir, '{}_gt.txt'.format(category))

    with open(pred, 'r') as f:
        preds = f.readlines()

    with open(gt, 'r') as f:
        gts = f.readlines()

    acc = sum(1 for x,y in zip(preds,gts) if x == y) / float(len(preds))
    all_acc.append(acc)

for i in range(len(all_acc)):
    print("category {}, acc = {}".format(category_list[i], all_acc[i]))

print("overall acc: ", sum(all_acc)/len(all_acc))

#python calculate_valid_acc_p3.py p3_log/results/
"""
category OP01-R02-TurkeySandwich, acc = 0.53557312253
category OP01-R04-ContinentalBreakfast, acc = 0.602851323829
category OP01-R07-Pizza, acc = 0.619587211655
category OP03-R04-ContinentalBreakfast, acc = 0.546681664792
category OP04-R04-ContinentalBreakfast, acc = 0.656221198157
category OP05-R04-ContinentalBreakfast, acc = 0.54746835443
category OP06-R03-BaconAndEggs, acc = 0.568665377176
('overall acc: ', 0.5824354646526777)
"""
