import argparse
import pickle

import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

import utils_mit_im as im_utils
import dataset_mit_states as data_utils

from vse_model import VSE
from vocabulary import Vocabulary
from dataset_coco import get_transform
from torch.utils import data
from torch.autograd import Variable
from PIL import Image

def main():
    print('evaluate vse on visual composition...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='runs/coco_vse++_restval/model_best.pth.tar')
    parser.add_argument('--data_root', default='data/mitstates_data')
    parser.add_argument('--image_data', default='mit_image_data.pklz')
    parser.add_argument('--labels_train', default='split_labels_train.pklz')
    parser.add_argument('--labels_test', default='split_labels_test.pklz')
    parser.add_argument('--meta_data', default='split_meta_info.pklz')
    parser.add_argument('--vocab_path', default='data/vocab')
    parser.add_argument('--crop_size', default=224)
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--visualize_object', default='dog')
    parser.add_argument('--visualize_attribute', default='red')

    args = parser.parse_args()
    print(args)

    imgdata = im_utils.load(args.data_root + '/' + args.image_data)
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
    dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=data_utils.custom_collate)

    writer = SummaryWriter()
    thumbsnailtrf = transforms.Compose([transforms.Scale(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])

    objembeddingslist = []
    imgtmblist = []
    objcounter = 0
    for i, (images, objatts, lengths, imgids, imgpaths) in enumerate(dataloader):
        if images is None:
            print 'None batch: full of unked'
            continue

        for j, objatt in enumerate(objatts):
            if objatt[2] == vocab(args.visualize_object):
                img = images[j].unsqueeze(0)
                imgemb = model.img_enc(Variable(img))
                objembeddingslist.append(imgemb)

                imgtmb = thumbsnailtrf(Image.open(imgpaths[j]).convert('RGB'))
                imgtmblist.append(imgtmb)

                objcounter += 1

    print('{} objects found and projected!'.format(objcounter))
    objembeddings = torch.cat(objembeddingslist, 0)
    imgthumbnails = torch.stack(imgtmblist, 0)

    writer.add_embedding(objembeddings, label_img=imgthumbnails)
    writer.close()

if __name__ == '__main__':
    main()