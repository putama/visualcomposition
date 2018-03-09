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
import torchvision.transforms as transforms

from operator import itemgetter
from PIL import Image, ImageDraw, ImageFont
from tensorboardX import SummaryWriter

def main():
    print('evaluate vse on visual composition...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='runs/composition_session4_allobjects_batch64_decay5/epoch_6_checkpoint.pth.tar')
    parser.add_argument('--data_root', default='data/mitstates_data')
    parser.add_argument('--image_data', default='mit_image_data.pklz')
    parser.add_argument('--labels_train', default='split_labels_train.pklz')
    parser.add_argument('--labels_test', default='split_labels_test.pklz')
    parser.add_argument('--meta_data', default='split_meta_info.pklz')
    parser.add_argument('--vocab_path', default='data/vocab')
    parser.add_argument('--crop_size', default=224)
    parser.add_argument('--batch_size', default=128)
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

    writer = SummaryWriter(log_dir='/'.join(args.model_path.split('/')[0:2]))
    pil2tensor = transforms.Compose([transforms.ToTensor()])

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

        targetdist = np.around(np.einsum('ij,ij->i', imgvecs, objattsvecs), decimals=5)

        allobjattdist = np.dot(imgvecs, allobjattsvecs.T)
        sorteddist = np.around(np.sort(allobjattdist), decimals=5) # handle numerical instability

        indices = np.argsort(allobjattdist[-1])[::-1]
        top3 = indices[0:3]

        topcaptions = map(lambda x: imgmetadata['pairNames'][x].lower(), top3)
        print imgpaths[-1].split('/')[-2] + ' => ' + str(topcaptions)

        pilimage = Image.open(imgpaths[-1]).resize((248,248))
        captimage = Image.new('RGB', pilimage.size, color = (255,255,255))
        drawcapt = ImageDraw.Draw(captimage)
        drawcapt.text((10,10), "Paired phrase:", fill=(0, 0, 0))
        drawcapt.text((10,30), '+ '+imgpaths[-1].split('/')[-2], fill=(0,0,0))
        drawcapt.text((10,50), "Retrieved phrase:", fill=(0, 0, 0))

        for counter, topcapt in enumerate(topcaptions):
            drawcapt.text((10, 10+(20*(counter+3))), '- '+topcapt, fill=(0,0,0))

        newimg = Image.new('RGB', (pilimage.size[0] * 2, pilimage.size[1]))
        newimg.paste(pilimage, (0, 0))
        newimg.paste(captimage, (pilimage.size[0], 0))
        writer.add_image('Image-'+str(i), pil2tensor(newimg), 0)

        top10pred = sum(targetdist >= sorteddist[:, -10])
        top5pred = sum(targetdist >= sorteddist[:, -5])
        top3pred = sum(targetdist >= sorteddist[:, -3])
        top2pred = sum(targetdist >= sorteddist[:, -2])
        top1pred = sum(targetdist >= sorteddist[:, -1])
        totaltop10 += top10pred
        totaltop5 += top5pred
        totaltop3 += top3pred
        totaltop2 += top2pred
        totaltop1 += top1pred

    print 'top-1 accuracy: {}, top-2 accuracy: {}, top-3 accuracy: {}'.format(
        totaltop1 * 100 / len(dataset), totaltop2 * 100 / len(dataset), totaltop3 * 100 / len(dataset))
    print 'top-5 accuracy: {}, top-10 accuracy: {}'.format(
        totaltop5 * 100 / len(dataset), totaltop10 * 100 / len(dataset))
    writer.close()

if __name__ == '__main__':
    main()
