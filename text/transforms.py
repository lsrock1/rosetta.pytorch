import torchvision.transforms as transforms
from torchvision.transforms import functional as F


def build_transforms(is_train):
    if is_train:
        transform_list = [
            Resize(is_train),
            transforms.RandomCrop((32, 128)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     [106.15212332443592, 104.32174808999343, 98.44559253254847],
            #     [6088.778646934532, 5757.71094035502, 5632.389237687235]
            # ),
            # transforms.Normalize(
            #     [69.3038496298514, 69.1912016384901, 69.2059632224818],
            #     [11806.07400489298, 11784.040819357813, 11782.340130056167]
            # ),
            # transforms.Normalize(
            #     [69.68162727546878/255, 69.55136871816666/255, 69.50573515782501/255],
            #     [11761.236612299039/255, 11734.780536034592/255, 11727.964664484216/255]
            # ),
            # transforms.Normalize(
            #     [ 41.73096137848201/255,  39.556890436298296/255, 38.89998827247813/255],
            #     [ 4109.207801317276/255,  3822.3808177967485/255,  3721.453936169103/255]
            # ),
            # transforms.ToPILImage(),
            # transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            # transforms.ToTensor()
        ]
    else:
        transform_list = [
            Resize(is_train),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     [106.15212332443592, 104.32174808999343, 98.44559253254847],
            #     [6088.778646934532, 5757.71094035502, 5632.389237687235]
            # ),
            # transforms.Normalize(
            #     [69.3038496298514, 69.1912016384901, 69.2059632224818],
            #     [11806.07400489298, 11784.040819357813, 11782.340130056167]
            # ),
            # transforms.Normalize(
            #     [ 41.73096137848201/255,  39.556890436298296/255, 38.89998827247813/255],
            #     [ 4109.207801317276/255,  3822.3808177967485/255,  3721.453936169103/255]
            # ),
            # transforms.ToTensor()
        ]
    return transforms.Compose(transform_list)

class Resize(object):
    def __init__(self, train):
        self.train = train

    def __call__(self, x):
        
        w, h = x.size
        
        if self.train:
            x = F.resize(x, (40, int((40 / h) * 1.2 * w)))
            width = x.size[0]
            if width > 136:
                x = F.resize(x, (40, 136))
            elif width < 136:
                x = F.pad(x, (0, 0, 136-width, 0))
        else:
            x = F.resize(x, (32, int((32 / h) * 1.2 * w)))
        return x

class Check(object):
    def __call__(self, x):
        print(len(x))