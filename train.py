import torch 
from dataset import FaceDataset
from model import StyleDiscriminator, Discriminator, AttentionDiscriminator
import config as cf
import torch.nn as nn
from torch.utils.data import DataLoader 
import torch.nn.functional as F  


if __name__ == '__main__':
    csv_file = '/home/nhatkhang/Documents/GANs/FakeReal/archive/test.csv'
    root_dir = '/home/nhatkhang/Documents/GANs/FakeReal/archive/real_vs_fake/real-vs-fake'
    faces = FaceDataset(csv_file, root_dir, cf.transform_b)

    classifier = Discriminator(3)

    trainset = DataLoader(faces, batch_size=2, shuffle=True)

    images, label = next(iter(trainset))

    print(images.shape)

    out1, out2 = classifier(images)
    print(label.shape)
    print(out1.shape)
    print(out2.shape)
    pix_label = torch.ones((2, 30,30))

    for i in range(2):
        pix_label[i] *= label[i]
    print(pix_label.shape)

    criterion = nn.BCEWithLogitsLoss()
    loss = F.binary_cross_entropy_with_logits(out1, label) + criterion(out2.squeeze(), pix_label)
    print(loss) 