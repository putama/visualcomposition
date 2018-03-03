import logging
import tensorboard_logger as tb_logger
import argparse

import time
import os
import pickle
import argparse
import torch
import numpy as np

import utils_mit_im as im_utils
import dataset_mit_states as data_utils

from vse_model import VSE
from vocabulary import Vocabulary
from dataset_coco import get_transform
from torch.utils import data
from torch.autograd import Variable
from vse_evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

from operator import itemgetter

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
    parser.add_argument('--crop_size', default=224)
    parser.add_argument('--batch_size', default=48)

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

    print('=> checkpoint loaded')
    print(opt)

    # construct model
    model = VSE(opt)
    # load model state
    model.load_state_dict(checkpoint['model'])

    print('=> VSE model initiated and weights loaded')

    # load mitstates dataset
    dataset = data_utils.MITstatesDataset(args.data_root, labelstest,
                                          imgdata, imgmetadata, vocab,
                                          transform=get_transform('train', opt))
    dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=data_utils.custom_collate)

    print("Forward CNN using ", torch.cuda.device_count(), " GPUs!")
    for epoch in range(opt.num_epochs):
        # average meters to record the training statistics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_logger = LogCollector()

        # switch to train mode
        model.train_start()

        end = time.time()
        for i, (images, objatts, lengths, imgids, imgpaths) in enumerate(dataloader):
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

        savepath = os.path.join(opt.logger_name,
                                'epoch_{}_checkpoint.pth.tar'.format(epoch+1))
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'opt': opt,
            'Eiters': model.Eiters,
        }, savepath)

if __name__ == '__main__':
    main()