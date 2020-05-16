import os
import torch
import parser
import data
import torch.nn as nn
import models

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def loop_iterable(iterable):
    while True:
        yield from iterable

def eval(feature_extractor, label_predictor, tar_test_loader):
    feature_extractor.eval(), label_predictor.eval()
    set_requires_grad(feature_extractor, requires_grad=False)
    set_requires_grad(label_predictor, requires_grad=False)

    total_accuracy = 0
    for x, y_true in iter(tar_test_loader):
        x, y_true = x.cuda(), y_true.cuda()
        y_feature = feature_extractor(x).view(x.shape[0], -1)
        y_pred = label_predictor(y_feature)
        total_accuracy += (y_pred.max(1)[1] == y_true.long()).float().mean().item()
    mean_accuracy = total_accuracy / len(tar_test_loader)

    return mean_accuracy

def main(args):
    src_feature_extractor, label_predictor, adda_discriminator = models.ADDA(args)
    src_feature_extractor.cuda(), label_predictor.cuda(), adda_discriminator.cuda()
    src_feature_extractor.load_state_dict(torch.load(os.path.join(args.src_model, 'source_feature_extractor.pt')))
    label_predictor.load_state_dict(torch.load(os.path.join(args.src_model, 'label_predictor.pt')))
    src_feature_extractor.eval(), label_predictor.eval()
    set_requires_grad(src_feature_extractor, requires_grad=False)
    set_requires_grad(label_predictor, requires_grad=False)

    tar_feature_extractor, _, _ = models.ADDA(args)
    tar_feature_extractor.cuda()
    tar_feature_extractor.load_state_dict(torch.load(os.path.join(args.src_model, 'source_feature_extractor.pt')))


    src_train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset=args.source_dataset),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    tar_train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset=args.target_dataset),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    tar_test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset=args.target_dataset),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)

    discriminator_optim = torch.optim.Adam(adda_discriminator.parameters(), lr=args.lr_adda_d)
    target_optim = torch.optim.Adam(tar_feature_extractor.parameters(), lr=args.lr_tar)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(src_train_loader), loop_iterable(tar_train_loader))
        tar_feature_extractor.train(), adda_discriminator.train()

        total_loss = 0
        total_accuracy = 0
        for _ in range(args.iterations):
            # Train discriminator
            set_requires_grad(tar_feature_extractor, requires_grad=False)
            set_requires_grad(adda_discriminator, requires_grad=True)
            for _ in range(args.k_disc):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.cuda(), target_x.cuda()

                source_features = src_feature_extractor(source_x).view(source_x.shape[0], -1)
                target_features = tar_feature_extractor(target_x).view(target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0]).cuda(),
                                             torch.zeros(target_x.shape[0]).cuda()])

                preds = adda_discriminator(discriminator_x).squeeze()
                loss = criterion(preds, discriminator_y)

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()

                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()

            # Train classifier
            set_requires_grad(tar_feature_extractor, requires_grad=True)
            set_requires_grad(adda_discriminator, requires_grad=False)
            for _ in range(args.k_clf):
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.cuda()
                target_features = tar_feature_extractor(target_x).view(target_x.shape[0], -1)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0]).cuda()

                preds = adda_discriminator(target_features).squeeze()
                loss = criterion(preds, discriminator_y)

                target_optim.zero_grad()
                loss.backward()
                target_optim.step()

        mean_loss = total_loss / (args.iterations*args.k_disc)
        mean_accuracy = total_accuracy / (args.iterations*args.k_disc)
        print(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, '
                   f'discriminator_accuracy={mean_accuracy:.4f}')

        # Create the full target model and save it
        acc = eval(tar_feature_extractor, label_predictor, tar_test_loader)
        print("accuracy: ", acc)
        if acc > best_acc:
            torch.save(tar_feature_extractor.state_dict(), os.path.join(args.save_dir, 'tar_feature_extractor.pt'))
            best_acc = acc


if __name__ == '__main__':
    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    main(args)
