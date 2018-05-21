import pickle
import argparse
import torch
import numpy as np
import time

import utils_mit_im as im_utils
import dataset_mit_states as data_utils

from vse_model import VSE
from vocabulary import Vocabulary
from dataset_coco import get_transform
from torch.utils import data
from torch.autograd import Variable
import torchvision.transforms as transforms

from operator import itemgetter
from PIL import Image, ImageDraw, ImageFont
from tensorboardX import SummaryWriter

def main():
    print('evaluate vse on visual composition...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='runs/composition_vse++/best.pth.tar')
    parser.add_argument('--data_root', default='data/mitstates_data')
    parser.add_argument('--image_data', default='mit_image_data.pklz')
    parser.add_argument('--labels_train', default='split_labels_train.pklz')
    parser.add_argument('--labels_test', default='split_labels_test.pklz')
    parser.add_argument('--meta_data', default='split_meta_info.pklz')
    parser.add_argument('--vocab_path', default='data/vocab')
    parser.add_argument('--crop_size', default=224)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--top_k', default=5)
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
                                 collate_fn=data_utils.collate_test)

    writer = SummaryWriter(log_dir='/'.join(args.model_path.split('/')[0:2]))
    pil2tensor = transforms.Compose([transforms.ToTensor()])

    allobjattsvecs = []
    for i, objatts in enumerate(dataset.get_all_pairs()):
        print '\r{} phrases encoded to vectors'.format(i * 64),
        if torch.cuda.is_available():
            objatts = objatts.cuda()
        objattsvecs = model.txt_enc(Variable(objatts), [4 for i in range(len(objatts))])
        allobjattsvecs.append(objattsvecs)
    allobjattsvecs = torch.cat(allobjattsvecs)

    top_k_meters = [AverageMeter() for k in range(args.top_k)]
    start_time = time.time()
    for i, (images, objatts, lengths, imgids, imgpaths, objatt_ids) in enumerate(dataloader):
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

        for k in range(args.top_k):
            num_correct = objatt_ids_correct[:,:k+1].contiguous().view(-1).sum()
            top_k_meters[k].update(num_correct*(100./args.batch_size), args.batch_size)

        print('\reval {:d}/{:d}: top@1.avg: {:.3f}, top@2.avg: {:.3f}, top@3.avg: {:.3f}, time: {:.3f}s'.format(
            (i+1) * args.batch_size, len(dataloader) * args.batch_size, top_k_meters[0].avg,
            top_k_meters[1].avg, top_k_meters[2].avg, (time.time() - start_time)
        )),
    #     targetdist = np.around(np.einsum('ij,ij->i', imgvecs, objattsvecs), decimals=5)
    #
    #     allobjattdist = np.dot(imgvecs, allobjattsvecs.T)
    #     sorteddist = np.around(np.sort(allobjattdist), decimals=5) # handle numerical instability
    #
    #     indices = np.argsort(allobjattdist[-1])[::-1]
    #     top3 = indices[0:3]
    #
    #     topcaptions = map(lambda x: imgmetadata['pairNames'][x].lower(), top3)
    #     print imgpaths[-1].split('/')[-2] + ' => ' + str(topcaptions)
    #
    #     pilimage = Image.open(imgpaths[-1]).resize((248,248))
    #     captimage = Image.new('RGB', pilimage.size, color = (255,255,255))
    #     drawcapt = ImageDraw.Draw(captimage)
    #     drawcapt.text((10,10), "Paired phrase:", fill=(0, 0, 0))
    #     drawcapt.text((10,30), '+ '+imgpaths[-1].split('/')[-2], fill=(0,0,0))
    #     drawcapt.text((10,50), "Retrieved phrase:", fill=(0, 0, 0))
    #
    #     for counter, topcapt in enumerate(topcaptions):
    #         drawcapt.text((10, 10+(20*(counter+3))), '- '+topcapt, fill=(0,0,0))
    #
    #     newimg = Image.new('RGB', (pilimage.size[0] * 2, pilimage.size[1]))
    #     newimg.paste(pilimage, (0, 0))
    #     newimg.paste(captimage, (pilimage.size[0], 0))
    #     writer.add_image('Image-'+str(i), pil2tensor(newimg), 0)

    # writer.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()