import parser
import torch
from torch.utils.data import DataLoader
import data
import os
import models


def do_epoch(feature_extractor, label_predictor, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    for x, y_true in iter(dataloader):
        x, y_true = x.cuda(), y_true.cuda()
        y_feature = feature_extractor(x).view(x.shape[0], -1)
        y_pred = label_predictor(y_feature)
        loss = criterion(y_pred, y_true.long())

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true.long()).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy


def main(args):
    src_train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset=args.source_dataset),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    src_test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset=args.source_dataset),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)

    feature_extractor, label_predictor, adda_discriminator = models.ADDA(args)
    feature_extractor.cuda(), label_predictor.cuda(), adda_discriminator.cuda()
    optim = torch.optim.Adam(list(feature_extractor.parameters()) + list(label_predictor.parameters()))
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        feature_extractor.train(), label_predictor.train()
        train_loss, train_accuracy = do_epoch(feature_extractor, label_predictor, src_train_loader, criterion, optim=optim)

        feature_extractor.eval(), label_predictor.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(feature_extractor, label_predictor, src_test_loader, criterion, optim=None)

        print(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(feature_extractor.state_dict(), os.path.join(args.save_dir, 'source_feature_extractor.pt'))
            torch.save(label_predictor.state_dict(), os.path.join(args.save_dir, 'label_predictor.pt'))

        lr_schedule.step(val_loss)


if __name__ == '__main__':
    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)
    main(args)
