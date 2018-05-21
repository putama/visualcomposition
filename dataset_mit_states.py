import os

import torch
import torch.utils.data as data
from PIL import Image

# TODO: performance gained by including all objects without attribute annotation
# include_all option in the constructor
class MITstatesDataset(data.Dataset):
    '''Custom dataset implementation of MITstates data'''
    def __init__(self, root, splitdata, imgdata, imgmetadata, vocab, transform=None, include_all=False):
        self.root = root
        self.splitdata = splitdata
        self.imgdata = imgdata
        self.imgmetadata = imgmetadata
        self.vocab = vocab
        self.transform = transform
        if include_all:
            self.imgids = self.splitdata['allTrainObImIds']
        else:
            self.imgids = self.splitdata['imIds']

    def __getitem__(self, index):
        # load image
        imgid = self.imgids[index]
        imgpath = self.imgdata['images'][imgid]['file_name']
        imgpathfull = os.path.join(self.root, 'images', imgpath.replace('_',' ',1))
        image = Image.open(imgpathfull).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # load att-obj phrase
        # check if attribute annotation if available
        if len(self.imgdata['annotations'][imgid]['pair_labs']) == 0:
            obj_id = self.imgdata['annotations'][imgid]['ob_labs'][0]
            obj_name = self.imgmetadata['objects'][obj_id]
            objatt_name = obj_name+'_<pad>'
            objatt_tensor = self.text_to_ids(objatt_name+obj_name)
            return image, objatt_tensor, None, imgid, imgpathfull, objatt_name
        else:
            objatt_id = self.imgdata['annotations'][imgid]['pair_labs'][0]
            objatt_name = self.imgmetadata['pairNames'][objatt_id]
            objatt_tensor = self.text_to_ids(objatt_name)
            return image, objatt_tensor, objatt_id, imgid, imgpathfull, objatt_name

    def text_to_ids(self, phrase):
        ids = [] # word/token ids
        # ids.append(self.vocab('<start>'))
        tokens = phrase.lower().split('_')
        for token in tokens:
            tokenid = self.vocab(token)
            ids.append(tokenid)
        # ids.append(self.vocab('<end>'))
        ids_tensor = torch.Tensor(ids)
        return ids_tensor

    def get_all_pairs(self):
        pairs = []
        for pairName in self.imgmetadata['pairNames']:
            pair = self.text_to_ids(pairName)
            if pair is not None:
                pairs.append(pair)

        group_size = 64
        grouppedpairs = []
        for i in range(0,len(pairs),group_size):
            grouppedpairs.append(torch.stack(pairs[i:i+group_size], 0).long())
        return grouppedpairs

    def __len__(self):
        return len(self.imgids)

def collate_train(items):
    items = filter(lambda x: x[0] is not None, items)

    # make sure items don't contain the same objatt_name
    objatt_names_set = set()
    items_distinct = []
    for item in items:
        if not item[5] in objatt_names_set:
            items_distinct.append(item)
            objatt_names_set.add(item[5])
    items = items_distinct

    images, objatt_tensors, objatt_ids, imgids, imgpaths, _ = zip(*items)
    lengths = [len(phrase) for phrase in objatt_tensors]

    # stack images and objatt phrase into a batch
    images = torch.stack(images, 0)
    objatt_tensors = torch.stack(objatt_tensors, 0).long()

    return images, objatt_tensors, lengths, imgids, imgpaths, objatt_ids

def collate_test(items):
    items = filter(lambda x: x[0] is not None, items)

    images, objatt_tensors, objatt_ids, imgids, imgpaths, _ = zip(*items)
    lengths = [len(phrase) for phrase in objatt_tensors]

    # stack images and objatt phrase into a batch
    images = torch.stack(images, 0)
    objatt_tensors = torch.stack(objatt_tensors, 0).long()

    return images, objatt_tensors, lengths, imgids, imgpaths, objatt_ids
