import sys

pred = sys.argv[1]
# path to p1_valid.txt file
gt = sys.argv[2]
# path to p1_gt.txt file


with open(pred, 'r') as f:
    preds = f.readlines()

with open(gt, 'r') as f:
    gts = f.readlines()

print("accuracy = ", sum(1 for x,y in zip(preds,gts) if x == y) / float(len(preds)))

#python calculate_valid_acc.py p1_log/p1_valid.txt p1_log/p1_gt.txt
# p1 acc: 0.436
