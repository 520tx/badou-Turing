import torchvision.transforms as T
def get_transform(height,width):

    #normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = T.Compose([
       # T.Resize((height, width)),
        # T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
        # T.RandomApply(
        # [T.ColorJitter(brightness=(0.3,0.6), contrast=(0.3,0.6), saturation=(0.3,0.6))], p = 0.5 
        # ),
        # T.RandomAffine(degrees=0.6, translate=(0.2,0.2), scale=(0.1,0.4))

    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
