import time
import os
import pickle
import torch
import logging
import tensorboard_logger as tb_logger
import argparse

import utils_mit_im as im_utils
import dataset_mit_states as data_utils

from vse_model import VSE
from vocabulary import Vocabulary
from dataset_coco import get_transform
from torch.utils import data
from torch.autograd import Variable
from vse_evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

# TODO: train from scratch
# TODO: fine tune the CNN
def main():
    print('parsing arguments')
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--vse_model_name', default='cocomit_vse++')
    parser.add_argument('--data_root', default='data/mitstates_data')
    parser.add_argument('--image_data', default='mit_image_data.pklz')
    parser.add_argument('--labels_train', default='split_labels_train.pklz')
    parser.add_argument('--labels_test', default='split_labels_test.pklz')
    parser.add_argument('--meta_data', default='split_meta_info.pklz')
    parser.add_argument('--vocab_path', default='data/vocab')
    parser.add_argument('--logger_name', default='runs/composition',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--lr_update', default=5, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--learning_rate', default=.0002, type=float)
    parser.add_argument('--finetune_cnn', action='store_true')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--top_k', default=5, type=int)

    args = parser.parse_args()
    print(args)

    imgdata = im_utils.load(args.data_root + '/' + args.image_data)
    labelstrain = im_utils.load(args.data_root + '/' + args.labels_train)
    labelstest = im_utils.load(args.data_root + '/' + args.labels_test)
    imgmetadata = im_utils.load(args.data_root + '/' + args.meta_data)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(args.logger_name, flush_secs=5)

    vse_model_path = os.path.join('runs', args.vse_model_name, 'model_best.pth.tar')

    # load model and options
    if torch.cuda.is_available():
        print('compute in GPU')
        checkpoint = torch.load(vse_model_path)
    else:
        print('compute in CPU')
        checkpoint = torch.load(vse_model_path, map_location=lambda storage, loc: storage)
    opt = checkpoint['opt']

    # load vocabulary used by the model
    with open('{}/{}_vocab.pkl'.format(args.vocab_path, opt.which_vocab), 'rb') as f:
        vocab = pickle.load(f)
        print('vocab loaded from: {}/{}_vocab.pkl'.format(args.vocab_path, opt.which_vocab))
    opt.vocab_size = len(vocab)

    # adjust the arguments
    opt.finetune = args.finetune_cnn
    opt.learning_rate = args.learning_rate
    print('=> checkpoint loaded')
    print(opt)

    # construct model
    model = VSE(opt)
    # load model state
    model.load_state_dict(checkpoint['model'])

    print('=> VSE model initiated and weights loaded')

    # load mitstates dataset
    dataset = data_utils.MITstatesDataset(args.data_root, labelstrain,
                                          imgdata, imgmetadata, vocab,
                                          transform=get_transform('train', opt))
    evaldataset = data_utils.MITstatesDataset(args.data_root, labelstest,
                                          imgdata, imgmetadata, vocab,
                                          transform=get_transform('test', opt))
    dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=data_utils.custom_collate)
    evaldataloader = data.DataLoader(dataset=evaldataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=data_utils.custom_collate)

    print("Forward CNN using ", torch.cuda.device_count(), " GPUs!")

    highestvalaverage = None
    for epoch in range(args.num_epochs):
        adjust_learning_rate(args, model.optimizer, epoch)

        # average meters to record the training statistics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_logger = LogCollector()

        # switch to train mode
        model.train_start()

        end = time.time()
        for i, (images, objatts, lengths, imgids, imgpaths, objatt_ids) in enumerate(dataloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # make sure train logger is used
            model.logger = train_logger
            model.train_emb(images, objatts, lengths)

            # Print log info
            if model.Eiters % opt.log_step == 0:
                logging.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        .format(
                        epoch, i, len(dataloader), batch_time=batch_time,
                        data_time=data_time, e_log=str(model.logger)))

            # Record logs in tensorboard
            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            tb_logger.log_value('step', i, step=model.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
            model.logger.tb_log(tb_logger, step=model.Eiters)

        _, currentvalaverage = eval_model(model, evaldataloader, top_k=args.top_k)
        if highestvalaverage is None:
            highestvalaverage = currentvalaverage
        if currentvalaverage >= highestvalaverage:
            modelpath = 'composition_best.pth.tar'
            print('saving model to: {}'.format(modelpath))
            savepath = os.path.join(args.logger_name, modelpath)
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'opt': opt
            }, savepath)
            highestvalaverage = currentvalaverage

def eval_model(model, dataloader, top_k=5):
    model.val_start()
    allobjattsvecs = []
    for i, objatts in enumerate(dataloader.dataset.get_all_pairs()):
        print '\r{} phrases encoded to vectors'.format(i * 64),
        if torch.cuda.is_available():
            objatts = objatts.cuda()
        objattsvecs = model.txt_enc(Variable(objatts), [4 for i in range(len(objatts))])
        allobjattsvecs.append(objattsvecs)
    allobjattsvecs = torch.cat(allobjattsvecs)

    top_k_meters = [AverageMeter() for k in range(top_k)]
    start_time = time.time()

    for i, (images, objatts, lengths, imgids, imgpaths, objatt_ids) in enumerate(dataloader):
        batch_size = images.size()[0]
        if images is None:
            print 'None batch: full of unked'
            continue

        objatt_ids = torch.LongTensor(objatt_ids).unsqueeze(1)
        if torch.cuda.is_available():
            images = images.cuda()
            objatt_ids = objatt_ids.cuda()

        # encode all images from test set
        imgvecs = model.img_enc(Variable(images, volatile=True))

        sorted_scores, sorted_predictions = torch.sort(imgvecs.matmul(allobjattsvecs.t()), 1, True)
        sorted_scores, sorted_predictions = sorted_scores.data, sorted_predictions.data
        objatt_ids = objatt_ids.expand_as(sorted_predictions)
        objatt_ids_correct = sorted_predictions.eq(objatt_ids)

        for k in range(top_k):
            num_correct = objatt_ids_correct[:, :k + 1].contiguous().view(-1).sum()
            top_k_meters[k].update(num_correct * (100. / batch_size), batch_size)

        if i % 100 == 0:
            print('\reval {:d}/{:d}: top@1.avg: {:.3f}, top@2.avg: {:.3f}, top@3.avg: {:.3f}, time: {:.3f}s'.format(
                (i + 1) * batch_size, len(dataloader) * batch_size, top_k_meters[0].avg,
                top_k_meters[1].avg, top_k_meters[2].avg, (time.time() - start_time)
            ))

    avgaccum = 0.
    for meter in top_k_meters:
        avgaccum += meter.avg
    avgall = avgaccum / len(top_k_meters)
    return top_k_meters, avgall

def adjust_learning_rate(opt, optimizer, epoch):
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()