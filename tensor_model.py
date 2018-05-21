import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np

from torch.utils import data
import dataset_mit_states as data_utils
from vocabulary import Vocabulary
import os
import argparse
import pickle
import utils_mit_im as im_utils
from dataset_coco import get_transform

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1).sqrt()
    norm = norm.unsqueeze(1) # recent version require non-singleton dimension for expand
    X = torch.div(X, norm.expand_as(X))
    return X

# tutorials/09 - Image Captioning
class EncoderImage(nn.Module):
    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            if not torch.cuda.is_available():
                self.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
                self.cnn.fc = nn.Sequential()
            else:
                self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
                self.cnn.module.fc = nn.Sequential()

        init_weights(self.fc)

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            if torch.cuda.is_available():
                model.cuda()
        else:
            if torch.cuda.is_available():
                model = nn.DataParallel(model).cuda()

        return model

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

class TensorEncoderText(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False, tensor_dim=100):
        super(TensorEncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        self.leakyRelu = nn.LeakyReLU(0.1)

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.embed.weight.data.uniform_(-0.1, 0.1)

        # projection down linear
        self.projdown_linear = nn.Linear(word_dim, tensor_dim)
        init_weights(self.projdown_linear)

        # projection up linear
        # self.projup_linear = nn.Linear(tensor_dim, self.embed_size)
        # init_weights(self.projup_linear)
        linlayer1 = nn.Linear(tensor_dim, self.embed_size / 2)
        linlayer2 = nn.Linear(self.embed_size / 2, self.embed_size)
        init_weights(linlayer1); init_weights(linlayer2)
        self.projup_linear = nn.Sequential(linlayer1, self.leakyRelu, linlayer2)

        # linear composition
        self.comp_linear = nn.Linear(tensor_dim*2, tensor_dim)
        init_weights(self.comp_linear)

        # tensor function parameter
        self.comp_tensor = nn.Parameter(torch.randn(tensor_dim, tensor_dim*2, tensor_dim*2))

    def forward(self, x, lengths):
        att_emb = self.embed(x[:,0])
        att_emb = self.projdown_linear(att_emb)
        att_emb = self.leakyRelu(att_emb)
        obj_emb = self.embed(x[:,1])
        obj_emb = self.projdown_linear(obj_emb)
        obj_emb = self.leakyRelu(obj_emb)

        cat_emb = torch.cat((att_emb, obj_emb), 1)
        slices = []
        for i in range(self.comp_tensor.size(0)):
            slice_element = cat_emb.matmul(self.comp_tensor[i]).mul(cat_emb).sum(1)
            slices.append(slice_element.view(-1,1))
        comp_term1 = torch.cat(slices, 1)
        comp_term2 = self.comp_linear(cat_emb)
        comp_vector = self.leakyRelu(comp_term1+comp_term2)
        out = self.projup_linear(comp_vector)
        out = l2norm(out)
        return out

    def load_embedding_weights(self, npwordvectors):
        wordvectors = torch.Tensor(npwordvectors)
        self.embed.weight.data.copy_(wordvectors)

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class TensorVSE(object):
    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.embed_size, opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs, no_imgnorm=opt.no_imgnorm)
        self.txt_enc = TensorEncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        # cuda version has slightly different structure
        if not torch.cuda.is_available():
            state_dict[0] = {key.replace(".module", ""): value for key, value in state_dict[0].items()}
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data[0], img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

def init_weights(linear):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(linear.in_features +
                              linear.out_features)
    linear.weight.data.uniform_(-r, r)
    linear.bias.data.fill_(0)