import torch
from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import json
import string

def get_label_mapping():
    return np.array(['Man', 'Woman', 'Person'])


class COCOGender(torch.utils.data.Dataset):
    def __init__(self, root, cfg, split='train', transform=None, size=None, compute_all=False):
        self.cfg = cfg
        self.original_root = os.path.expanduser(root)
        self.transform  = transform
        self.split      = split
        self.root       = os.path.join(self.original_root, 'COCO')
        self.image_root = os.path.join(
            self.original_root,
            'COCO',
            '{}2014'.format(
                'train' if split == 'train' else 'val'
            )
        )
        self.size                  = cfg.DATA.SIZE
        self.remove_background     = cfg.DATA.REMOVE_BACKGROUND
        self.return_attention      = cfg.DATA.ATTENTION_DIR != "NONE"
        self.min_needed            = cfg.DATA.MIN_NEEDED
        self.label_eval_like_train = cfg.DATA.LABEL_EVAL_LIKE_TRAIN
        self.return_attention      = cfg.DATA.ATTENTION_DIR != "NONE"

        self.seg_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.size,self.size)),
                transforms.ToTensor(),
            ]
        )

        # Binary: Only two classes - man, woman
        if split == 'train':
            self.binary = self.cfg.DATA.BINARY_TRAIN
        else:
            self.binary = self.cfg.DATA.BINARY_EVAL

        with open(os.path.join(self.root, 'COCO_gender', 'captions_only_valtrain2014.json')) as f:
            captions = json.load(f)
        captions = captions['annotations']
        id2captions = {}
        for entry in captions:
            cap_id = entry['image_id']
            if cap_id not in id2captions:
                id2captions[cap_id] = []
            id2captions[cap_id].append(entry['caption'])
        self.id2captions = id2captions

        # For convenience, option to compute all filenames (train, val, and test)
        if compute_all:
            train_coco = COCO('data/COCO/annotations/instances_train2014.json')
            train_filenames, train_labels, train_ids = self.load_filenames_labels(train_coco, 'train')
            val_coco = COCO('data/COCO/annotations/instances_val2014.json')
            val_filenames,   val_labels, val_ids  = self.load_filenames_labels(val_coco, 'val')
            test_filenames, test_labels, test_ids = self.load_filenames_labels(val_coco, 'test')
            self.data = np.concatenate((train_filenames, val_filenames, test_filenames))
            self.all_labels = torch.cat((train_labels, val_labels, test_labels))
        else:
            self.data = None

        coco = COCO(
            'data/COCO/annotations/instances_{}2014.json'.format(
                'train' if split == 'train' else 'val'
            )
        )

        self.filenames, self.labels, self.ids = self.load_filenames_labels(coco, split)

        if self.binary:
            # Filter filenames and labels for only man, woman classes (remove person class)
            binary_inds = torch.where(self.labels <= 1)[0]
            self.filenames = self.filenames[binary_inds]
            self.labels    = self.labels[binary_inds]
            self.ids       = self.ids[binary_inds]

        if self.return_attention:
            self.attention_data = np.array(
                [
                    os.path.join(
                        self.root,
                        'COCO_gender',
                        cfg.DATA.ATTENTION_DIR,
                        path.split(self.root)[-1].strip(os.sep).replace('.jpg', '.pth')) for path in self.filenames
                ]
            )

        print('{} STATS:'.format(split.upper()))
        print('NUM MAN:    {}'.format(len(torch.where(self.labels==0)[0])))
        print('NUM WOMAN:  {}'.format(len(torch.where(self.labels==1)[0])))
        print('NUM PERSON: {}'.format(len(torch.where(self.labels==2)[0])))

        # Load bounding boxes and segmentation
        self.masks = []
        self.bbox_coords = []
        coco_entries = coco.loadImgs(self.ids.astype(int))
        for entry in coco_entries:
            person_cat = 1
            annIds = coco.getAnnIds(imgIds=entry['id'], catIds=[person_cat], iscrowd=None)
            anns = coco.loadAnns(annIds)
            if len(anns) == 0:
                # "Ignore" mask is all zeros.
                self.masks.append(torch.zeros(1,1,1))
                self.bbox_coords.append(np.zeros(4))
            else:
                anns = find_largest_bbox(anns)
                mask = coco.annToMask(anns[0])
                mask = torch.Tensor(mask).unsqueeze(0)
                self.masks.append(mask)
                self.bbox_coords.append(anns[0]['bbox'])


    def load_filenames_labels(self,coco, split):
        root = os.path.join(
            self.original_root,
            'COCO',
            '{}2014'.format(
                'train' if split == 'train' else 'val'
            )
        )
        if split == 'train':
            train_ids = np.loadtxt(
                os.path.join(self.root, 'COCO_gender', 'biased_split', 'train.ids.txt'),
                dtype=float
            )
            filenames, labels, data_ids = filter_data(
                coco, train_ids, root, self.min_needed, self.id2captions
            )

        elif split == 'val':
            val_man   = np.loadtxt(
                os.path.join(self.root, 'COCO_gender', 'balanced_split', 'val_man.txt'),
                dtype=float
            )
            val_woman = np.loadtxt(
                os.path.join(self.root, 'COCO_gender', 'balanced_split', 'val_woman.txt'),
                dtype=float
            )
            data_ids  = np.concatenate((val_man, val_woman))
            if self.label_eval_like_train:
                filenames, labels, data_ids = filter_data(
                    coco, data_ids, root, self.min_needed, self.id2captions
                )

            else:
                filenames = get_filenames(coco, data_ids, root)
                labels    = np.concatenate((np.zeros(len(val_man)), np.ones(len(val_woman))))
                labels    = torch.Tensor(labels)

        else:
            assert split == 'test'
            test_man   = np.loadtxt(
                os.path.join(self.root, 'COCO_gender', 'balanced_split', 'test_man.txt'),
                dtype=float
            )
            test_woman = np.loadtxt(
                os.path.join(self.root, 'COCO_gender', 'balanced_split', 'test_woman.txt'),
                dtype=float
            )
            data_ids   = np.concatenate((test_man, test_woman))
            if self.label_eval_like_train:
                filenames, labels, data_ids = filter_data(
                    coco, data_ids, root, self.min_needed, self.id2captions
                )
            else:
                labels     = np.concatenate((np.zeros(len(test_man)), np.ones(len(test_woman))))
                labels     = torch.Tensor(labels)
                filenames  = get_filenames(coco, data_ids, root)

        return filenames, labels, data_ids


    def __getitem__(self, index):
        path = self.filenames[index]
        img = Image.open(path).convert('RGB')
        label = self.labels[index]

        # Bounding box
        bbox = self.bbox_coords[index]
        if np.all(bbox) == 0:
            # Make bbox_im all zeros
            bbox_im = torch.zeros((1,self.size,self.size))
        else:
            bbox = np.round(bbox).astype(int)
            arr = np.array(img)
            bbox_im = np.zeros(arr.shape[:2])
            bbox_im[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0]+bbox[2])] = 1
            bbox_im = torch.Tensor(bbox_im).unsqueeze(0).unsqueeze(0)
            bbox_im = F.interpolate(
                bbox_im, size=(self.size, self.size), mode='bilinear',
                align_corners=False)[0]

        # Segmentation
        seg = self.masks[index]
        seg = self.seg_transform(seg)

        if self.transform is not None:
            img = self.transform(img)
            if self.remove_background:
               img = img * seg
        if self.return_attention:
            att = torch.load(self.attention_data[index])
            if 'deeplab' in self.cfg.DATA.ATTENTION_DIR:
                att = att['mask']
            else:
                att = att['unnormalized_attentions']
            att = F.interpolate(att, size=(self.size, self.size), mode='bilinear',
                                align_corners=False)[0]

        else:
            att  = torch.Tensor([-1]) # So batches correctly

        return {
            'image_path': path,
            'image':       img,
            'label':     label,
            'seg':         seg,
            'bbox':    bbox_im,
            'attention': att
        }

    def __len__(self):
        return len(self.filenames)



class ImageInfo():
    def __init__(self, coco_entry, caption_list):
        self.filename = coco_entry['file_name']
        self.id = coco_entry['id']
        self.captions = caption_list
        self.num_man      = 0
        self.num_woman    = 0
        self.num_person   = 0
        self.num_else     = 0 # no man, woman, or person words
        self.num_multiple = 0 # caption mentions multiple categories of people



def get_filenames(coco, ids, image_root):
    """
    coco: pycocotools.coco.COCO object
    ids: numpy array of COCO dataset ids
    image_root: path to COCO image folder

    Return numpy array of filenames
    """
    coco_entries = coco.loadImgs(ids.astype(int))
    filenames = []
    for c in coco_entries:
        filenames.append(os.path.join(image_root, c['file_name']))
    return np.array(filenames)


def filter_data(coco, ids, image_root, min_needed, id2captions):
    """
    coco: pycocotools.coco.COCO object
    ids: numpy array of COCO dataset ids
    image_root: path to COCO image folder
    min_needed: int between [1,5]. Minimum number of captions that need to mention
        a gendered word to be counted as that class. For class 'person', image needs to
        first not qualify as man or woman, and also need to have at least min_needed
        'person' synonyms.
    id2captions: dictionary which maps image ids to list of captions

    Returns:
    filenames: numpy array, contains filenames of images which qualified as either
        man, woman, or person based on min_needed.
    labels: torch tensor, same length as filenames. Corresponding labels - 0 = man, 1 = woman, 2 = person
    ids:    numpy array, contains image ids, used for later processing of data
           (like getting bounding boxes & masks)
    """

    id2info = {}
    coco_entries = coco.loadImgs(ids.astype(int))

    for entry in coco_entries:
        assert entry['id'] not in id2info
        info = ImageInfo(entry, id2captions[entry['id']])
        counts = count_man_woman_person(info.captions)
        info.num_man      = counts['num_man']
        info.num_woman    = counts['num_woman']
        info.num_person   = counts['num_person']
        info.num_else     = counts['num_else']
        info.num_multiple = counts['num_multiple']

        # Only include images where the captions don't conflict on man/woman,
        # and captions that include at least one person mentioned
        if info.num_multiple == 0 and \
            (info.num_man + info.num_woman + info.num_person) > 0:
            id2info[entry['id']] = info

    filenames = []
    labels    = []
    ids       = []
    for info in id2info.values():
        assert info.num_multiple == 0 and (info.num_man + info.num_woman + info.num_person) > 0
        if info.num_man >= min_needed:
            filenames.append(os.path.join(image_root, info.filename))
            labels.append(0)
            ids.append(info.id)
        elif info.num_woman >= min_needed:
            filenames.append(os.path.join(image_root, info.filename))
            labels.append(1)
            ids.append(info.id)
        elif info.num_person >= min_needed:
            filenames.append(os.path.join(image_root, info.filename))
            labels.append(2)
            ids.append(info.id)
    return np.array(filenames), torch.Tensor(labels), np.array(ids)


def find_largest_bbox(annotations):
    """
    annotations: list of coco-style annotations.

    if only one annotation, return annotations.
    if more than one, find annotation that has largest
    bounding box by area, and return list of length 1
    containing only that annotation.
    """
    if len(annotations) == 1:
        return annotations
    else:
        max_area   = -1
        best_index = -1
        for i,annot in enumerate(annotations):
            area = annot['bbox'][2] * annot['bbox'][3]
            if area > max_area:
                max_area = area
                best_index = i
        return [annotations[best_index]]


man_word_list_synonyms = ['boy', 'brother', 'dad', 'husband', 'man',       \
        'groom', 'male','guy', 'dude', 'policeman', 'father',              \
        'son', 'fireman', 'actor','gentleman', 'boyfriend',                \
        'mans', 'his', 'obama', 'businessman', 'he', 'cowboy']
woman_word_list_synonyms = ['girl', 'sister', 'mom', 'wife', 'woman',      \
        'bride', 'female', 'lady',  'actress', 'nun', 'girlfriend',        \
        'her', 'she', 'mother', 'daughter', 'businesswoman', 'cowgirl']
person_word_list_synonyms = ['person', 'child', 'kid', 'teenager',         \
        'someone', 'player', 'rider',  'skiier','chef',                    \
        'snowboarder', 'surfer',  'hipster',   'skateboarder', 'adult',    \
        'baby', 'skier', 'diver', 'bicycler', 'hiker', 'student',          \
        'shopper', 'cyclist', 'officer', 'teen', 'worker', 'passenger',    \
        'cook', 'pedestrian', 'employee',  'driver', 'skater',              \
        'toddler',  'fighter', 'patrol', 'cop', 'server', 'carrier',       \
        'player', 'motorcyclist', 'carpenter', 'owner',  'individual',     \
        'bicyclist', 'boarder',  'boater', 'painter', 'artist',            \
        'citizen', 'staff', 'biker', 'technician', 'hand',  'baker',       \
        'manager', 'plumber', 'hands', 'performer', 'rollerblader',        \
        'farmer', 'athlete', 'pitcher', 'soldier']

def count_man_woman_person(caption_list):
    exclude = set(string.punctuation)

    num_man = 0
    num_woman = 0
    num_person = 0
    num_else = 0
    num_multiple = 0
    for c in caption_list:
        # Lowercase and remove punctuation
        c = c.lower()
        c = ''.join(ch for ch in c if ch not in exclude)
        c = c.split(' ')

        man, woman, person = 0,0,0
        for w in man_word_list_synonyms:
            if w in c:
                man = 1
        for w in woman_word_list_synonyms:
            if w in c:
                woman = 1
        for w in person_word_list_synonyms:
            if w in c:
                person = 1

        if (man + woman) > 1:
            num_multiple += 1
        elif (man + woman + person) == 0:
            num_else += 1
            #print('Caption with no people words: {}'.format(' '.join(c)))
        else:
            num_man += man
            num_woman += woman
            num_person += person

    return {
        'num_man': num_man,
        'num_woman': num_woman,
        'num_person': num_person,
        'num_else': num_else,
        'num_multiple': num_multiple
    }



def get_loss_upweights(min_needed, binary=False, only_woman=False):
    """
    For weighting training loss for imbalanced classes.
    # training samples for man > woman > person, so upweight
    woman & person classes. (Or just woman, if binary classification)

    min_needed: int, specifies which dataset setting is being used.
      (min # captions with man/woman/person words to qualify
       as that class)
    binary: only have man & woman classes
    only_woman: only upweight woman. keep man and person weight as 1.

    Returns 1D tensor of length 3 (or 2 for binary), with loss rescaling weights,
    in order [man, woman, person] (or [man, woman], in accordance
    with order of classes.

    weight w_c for class c in C is calculated as:
    (1 / num_samples_in_class_c) / (max(1/num_samples_in_c) for c in num_classes)

    """

    # Maps min_needed to stats
    training_dataset_stats = {
        1: {
            'man': 16246,
            'woman': 6245,
            'person': 15,
            'total': 22506
        },
        2: {
            'man': 13218,
            'woman': 5456,
            'person': 1691,
            'total': 20365
        },
        3: {
            'man': 10565,
            'woman': 4802,
            'person': 2822,
            'total': 18189
        },
        4: {
            'man': 7761,
            'woman': 3967,
            'person': 1316,
            'total': 13044
        },
        5: {
            'man': 4246,
            'woman': 2371,
            'person':102,
            'total': 6719
        }
    }

    stats   = training_dataset_stats[min_needed]
    if only_woman:
        weights = torch.Tensor([1., stats['man']/stats['woman'], 1.])
        if binary:
            weights = weights[:2]
    else:
        counts  = torch.Tensor([stats['man'], stats['woman'], stats['person']])
        if binary:
            counts = counts[:2]
        fracs   = 1 / counts
        weights = fracs / torch.max(fracs)

    return weights





