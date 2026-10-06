import torchvision.transforms as T


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Construct training image transformations with domain-appropriate data augmentations.
    Augmentations are strictly applied only during training to enhance generalization.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """
    Construct deterministic evaluation image transformations for validation, testing, and inference.
    No random augmentations are performed during evaluation.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
