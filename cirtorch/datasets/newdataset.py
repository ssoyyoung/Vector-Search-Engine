import numpy as np
import os
import glob

from PIL import Image
from collections import defaultdict

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

from torchvision import transforms
from tqdm import tqdm

def path2label(path):
    return '-'.join(path.split('/')[-3:-1])

class Triplet(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset, train, transform):
        self.dataset = dataset
        self.train = train
        self.transform = transform
        self.queue = []
        self.qsize = 9999999

        if self.dataset.startswith('aliexpress'):
            data_root = '/home/piclick/github/cgd/cgd.crawling/images/'
            shops = glob.glob(data_root + '*/*/*/query*')
            feedbacks = glob.glob(data_root + '*/*/*/fb*')
            labels = {path2label(q):i for i, q in enumerate(shops)}
           
            if self.train:
                print("Generate Train Queue ...")
                for index in tqdm(range(len(labels)), total=len(labels)):
                    q = shops[index]
                    qlabel = labels[path2label(q)]
                    if qlabel == 0: continue

                    positives = glob.glob(q.replace(q.split('/')[-1],'fb*'))

                    if len(positives) == 0:
                        labels.pop(path2label(q))
                        continue

                    for p in positives:
                        n = np.random.choice(feedbacks)
                        while path2label(n) == path2label(q):
                            n = np.random.choice(feedbacks)
                        try:
                            nlabel = labels[path2label(n)]
                        except KeyError:
                            continue

                        self.queue.append([(q,p,n),(qlabel,qlabel,nlabel)])

                    if len(self.queue) > self.qsize:
                        break

                print('')
                self.train_data = self.queue
                labels = {k:i for i,(k,v) in enumerate(labels.items())}

                print("Total Queue :", len(self.queue))
                print("Total Labels :", len(labels.keys()))

        elif self.dataset.startswith('cub'):
            data_root = '/data/cub200-2011/CUB_200_2011/'
            img_root = os.path.join(data_root, 'images')
            images_file = os.path.join(data_root, 'images.txt')
            train_test_split = os.path.join(data_root, 'train_test_split.txt')
            image_class_labels = os.path.join(data_root, 'image_class_labels.txt')

            trainIdx =[]
            testIdx = []

            with open(train_test_split) as tts:
                for i,line in enumerate(tts.readlines()):
                    if line.split(' ')[-1].replace('\n','') == '1': trainIdx.append(i)
                    else: testIdx.append(i)

            with open(images_file) as f:
                images = [os.path.join(img_root,line.split(' ')[-1].replace('\n','')) for line in f.readlines()]

            with open(image_class_labels) as f:
                image_class_labels = [(line.split(' ')[0], line.split(' ')[-1].replace('\n','')) for line in f.readlines()]

            self.images = images

            if self.train:
                # train_image_class_labels = [image_class_labels[i] for i in trainIdx]
                # self.train_data = [images[i] for i in trainIdx]
                # self.train_labels = [image_class_labels[i][1] for i in trainIdx]

                train_image_class_labels = image_class_labels
                self.train_data = images
                self.train_labels = [image_class_labels[i][1] for i in range(len(image_class_labels))]

                label_to_indices = defaultdict(list)
                for index, label in train_image_class_labels: 
                    label_to_indices[label].append(index)
                
                self.label_to_indices = dict(label_to_indices)
                self.labels_set = set([label for index, label in train_image_class_labels])

            else:
                self.test_data = [images[i] for i in testIdx]
                self.test_labels = [image_class_labels[i][1] for i in testIdx]

    def __getitem__(self, index):
        if self.dataset.startswith('aliexpress'):
            if self.train:
                imgs, labels = self.queue[index]
                label1, label1, negative_label = labels

                img1 = Image.open(imgs[0]).convert('RGB')
                img2 = Image.open(imgs[1]).convert('RGB')
                img3 = Image.open(imgs[2]).convert('RGB')

                imgs = [np.asarray(img1), np.asarray(img2), np.asarray(img3)]
                
                for idx, img in enumerate(imgs):
                    if len(img.shape) == 2:
                        img = np.stack((img,) * 3, axis=-1)
                        imgs[idx] = Image.fromarray(img)
                    else:                
                        imgs[idx] = Image.fromarray(img)

                if self.transform is not None:
                    img1 = self.transform(imgs[0])
                    img2 = self.transform(imgs[1])
                    img3 = self.transform(imgs[2])

        elif self.dataset.startswith('cub'):
            if self.train:
                img1, label1 = self.train_data[index], self.train_labels[index]
                positive_index = index
                while positive_index == index:
                    positive_index = int(np.random.choice(self.label_to_indices[label1]))
                negative_label = np.random.choice(list(self.labels_set - set([label1])))
                negative_index = int(np.random.choice(self.label_to_indices[negative_label]))

                img2 = self.train_data[positive_index]
                img3 = self.train_data[negative_index]
            else:
                img1 = self.test_data[self.test_triplets[index][0]]
                img2 = self.test_data[self.test_triplets[index][1]]
                img3 = self.test_data[self.test_triplets[index][2]]

            img1 = Image.open(img1)
            img2 = Image.open(img2)
            img3 = Image.open(img3)
            
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)

        # q, p, n
        return (img1, img2, img3), [label1, label1, negative_label]

    def __len__(self):
        return len(self.train_data)