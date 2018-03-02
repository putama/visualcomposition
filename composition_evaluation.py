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

from operator import itemgetter


def main():
    print('evaluate vse on visual composition...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='runs/cocomit_vse++/model_best.pth.tar')
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

    # load model params checkpoint and options
    if torch.cuda.is_available():
        print('compute in GPU')
        checkpoint = torch.load(args.model_path)
    else:
        print('compute in CPU')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
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

    print('=> model initiated and weights loaded')

    # load mitstates dataset
    dataset = data_utils.MITstatesDataset(args.data_root, labelstest,
                                          imgdata, imgmetadata, vocab,
                                          transform=get_transform('test', opt))
    dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=data_utils.custom_collate)

    allobjattsvecs = []
    for i, objatts in enumerate(dataset.get_all_pairs()):
        print '{} phrases encoded to vectors'.format(i * 64)
        if torch.cuda.is_available():
            objatts = objatts.cuda()
        objattsvecs = model.txt_enc(Variable(objatts), [4 for i in range(len(objatts))])
        allobjattsvecs.append(objattsvecs)
    allobjattsvecs = torch.cat(allobjattsvecs)
    allobjattsvecs = allobjattsvecs.data.cpu().numpy()

    totaltop10 = 0.0
    totaltop5 = 0.0
    totaltop3 = 0.0
    totaltop2 = 0.0
    totaltop1 = 0.0
    correctpaths = []
    for i, (images, objatts, lengths, imgids, imgpaths) in enumerate(dataloader):
        if images is None:
            print 'None batch: full of unked'
            continue
        print '{}/{} data items encoded'.format(i * args.batch_size, len(dataloader) * args.batch_size)

        if torch.cuda.is_available():
            objatts = objatts.cuda()
            images = images.cuda()

        # encode all attribute-object pair phrase on test set
        objattsvecs = model.txt_enc(Variable(objatts), lengths)
        objattsvecs = objattsvecs.data.cpu().numpy()
        # encode all images from test set
        imgvecs = model.img_enc(Variable(images))
        imgvecs = imgvecs.data.cpu().numpy()

        targetdist = np.einsum('ij,ij->i', imgvecs, objattsvecs)
        allobjattdist = np.dot(imgvecs, allobjattsvecs.T)
        sorteddist = np.sort(allobjattdist)
        top10pred = sum(targetdist > sorteddist[:, -10])
        top5pred = sum(targetdist > sorteddist[:, -5])
        top3pred = sum(targetdist > sorteddist[:, -3])
        top2pred = sum(targetdist > sorteddist[:, -2])
        top1pred = sum(targetdist > sorteddist[:, -1])
        totaltop10 += top10pred
        totaltop5 += top5pred
        totaltop3 += top3pred
        totaltop2 += top2pred
        totaltop1 += top1pred
        if top1pred > 0:
            foundpaths = itemgetter(*np.where(targetdist > sorteddist[:, -1])[0])(imgpaths)
            if isinstance(foundpaths, basestring):
                correctpaths.append(foundpaths)
            else:
                correctpaths.extend(foundpaths)

    print 'top-1 accuracy: {}, top-2 accuracy: {}, top-3 accuracy: {}'.format(
        totaltop1 * 100 / len(dataset), totaltop2 * 100 / len(dataset), totaltop3 * 100 / len(dataset))
    print 'top-5 accuracy: {}, top-10 accuracy: {}'.format(
        totaltop5 * 100 / len(dataset), totaltop10 * 100 / len(dataset))
    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
