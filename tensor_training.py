import torch
import logging
import tensorboard_logger as tb_logger
import argparse

import pickle
import os
import time
import shutil

import utils_mit_im as im_utils
import dataset_mit_states as data_utils

from torch.autograd import Variable
from dataset_coco import get_transform
from torch.utils import data
import vocabulary
from vocabulary import Vocabulary
from vse_model import VSE
from tensor_model import TensorVSE
from mlp_model import MlpVSE
from vse_evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

# TODO train without start & end tokens --
# TODO make sure each batch don't contain examples from the same class --
# TODO include attributeless object --
# TODO train for several epoch before finetune the network --
# TODO image crop size and pixel mean for normalization
# TODO grad clip and xavier initialization
# TODO train only on object or attribute
# TODO analysis of what goes right or wrong
# TODO suggest better composition functions

# TODO writing: discuss about the arithmatical properties of the shared space
def main():
    print('parsing arguments')
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/mitstates_data')
    parser.add_argument('--image_data', default='mit_image_data.pklz')
    parser.add_argument('--labels_train', default='split_labels_train.pklz')
    parser.add_argument('--labels_test', default='split_labels_test.pklz')
    parser.add_argument('--meta_data', default='split_meta_info.pklz')
    parser.add_argument('--vocab_path', default='./data/vocab',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--tensor_dim', default=100, type=int,
                        help='Dimensionality of the tensor function')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=50, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='runs/rnn_finetune',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='resnet50',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--which_vocab', default='cocomit',
                        help='{cocomit|coco|fasttext} choose either coco or fasttext word vectors')
    parser.add_argument('--composition_function', default='tensor', help='tensor|rnn')
    parser.add_argument('--top_k', default=3, type=int)
    parser.add_argument('--use_all_obj', action='store_true',
                        help='use all objects for training including ones without attribute')

    # multiple gpu training
    parser.add_argument('--gpus', default=[0, 1], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")

    opt = parser.parse_args()
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab_full_path = os.path.join(opt.vocab_path, '%s_vocab.pkl' % opt.which_vocab)
    if not os.path.isfile(vocab_full_path):  # build the vocab in case the file not exists
        vocabulary.run(opt.data_path, opt.which_vocab)
    vocab = pickle.load(open(vocab_full_path, 'rb'))
    opt.vocab_size = len(vocab)

    if opt.composition_function == 'tensor':
        model = TensorVSE(opt)
        print('Tensor Composition model initiated')
    elif opt.composition_function == 'mlp':
        model = MlpVSE(opt)
        print('MLP Composition T model initiated')
    else:
        model = VSE(opt)
        print('RNN-based VSE model initiated')

    # initilize the word embeddings if cocomit vocab is chosen
    if opt.which_vocab == 'cocomit':
        vectors_full_path = os.path.join('./data/%s/%s_vectors.pkl' % ('fasttext', opt.which_vocab))
        print("=> loading from vector file: " + vectors_full_path)
        # read the vectors in case it is not exist
        if not os.path.isfile(vectors_full_path):
            vocabulary.run(opt.data_path, opt.which_vocab)
        wordvectors = pickle.load(open(vectors_full_path, 'rb'))
        model.txt_enc.load_embedding_weights(wordvectors)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint['model'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # load mitstates dataset
    imgdata = im_utils.load(opt.data_root + '/' + opt.image_data)
    labelstrain = im_utils.load(opt.data_root + '/' + opt.labels_train)
    labelstest = im_utils.load(opt.data_root + '/' + opt.labels_test)
    imgmetadata = im_utils.load(opt.data_root + '/' + opt.meta_data)
    dataset = data_utils.MITstatesDataset(opt.data_root, labelstrain,
                                          imgdata, imgmetadata, vocab,
                                          transform=get_transform('train', opt), include_all=opt.use_all_obj)
    evaldataset = data_utils.MITstatesDataset(opt.data_root, labelstest,
                                              imgdata, imgmetadata, vocab,
                                              transform=get_transform('test', opt))
    dataloader = data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True,
                                 collate_fn=data_utils.collate_train)
    evaldataloader = data.DataLoader(dataset=evaldataset, batch_size=opt.batch_size, shuffle=False,
                                     collate_fn=data_utils.collate_test)

    print("Forward CNN using ", torch.cuda.device_count(), " GPUs!")

    highestvalaverage = None
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

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
                    'Batch size: {3}\t'
                    '{e_log}\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        .format(epoch, i, len(dataloader), images.size(0),
                                data_time=data_time, e_log=str(model.logger)))

            # Record logs in tensorboard
            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            tb_logger.log_value('step', i, step=model.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
            model.logger.tb_log(tb_logger, step=model.Eiters)

        _, currentvalaverage = eval_model(epoch, model, evaldataloader,
                                          top_k=opt.top_k, log_dir=opt.logger_name)
        if highestvalaverage is None:
            highestvalaverage = currentvalaverage
        if currentvalaverage >= highestvalaverage:
            modelpath = 'composition_best.pth.tar'
            print('saving model to: {}'.format(modelpath))
            savepath = os.path.join(opt.logger_name, modelpath)
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'opt': opt
            }, savepath)
            highestvalaverage = currentvalaverage

def eval_model(epoch, model, dataloader, top_k=3, log_dir=''):
    model.val_start()
    allobjattsvecs = []
    for i, objatts in enumerate(dataloader.dataset.get_all_pairs()):
        print '\r{} phrases encoded to vectors'.format(i * 64),
        if torch.cuda.is_available():
            objatts = objatts.cuda()
        objattsvecs = model.txt_enc(Variable(objatts), [objatts.size(1) for i in range(len(objatts))])
        allobjattsvecs.append(objattsvecs)
    allobjattsvecs = torch.cat(allobjattsvecs)

    top_k_meters = [AverageMeter() for k in range(top_k)]
    start_time = time.time()

    with open(os.path.join(log_dir, 'traineval_log.txt'), 'a') as logfile:
        header = 'evaluate model epoch: {}\n'.format(epoch+1)
        logfile.write(header)
        print header

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

        if (i > 0 and i % 100 == 0) or i == len(dataloader)-1:
            with open(os.path.join(log_dir, 'traineval_log.txt'), 'a') as logfile:
                body = '{:d}/{:d}: top@1.avg: {:.3f}, top@2.avg: {:.3f}, top@3.avg: {:.3f}, time: {:.3f}s\n'.format(
                    i, len(dataloader)-1, top_k_meters[0].avg, top_k_meters[1].avg,
                    top_k_meters[2].avg, (time.time() - start_time))
                logfile.write(body)
                print(body)


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
    torch.manual_seed(99)
    main()