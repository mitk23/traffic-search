import torch

from config import BATCH_SIZE, RANDOM_SEED, SPACE_WINDOW, TIME_STEP
from dataset import STDataset
from storage import X_TEST, X_TRAIN, Y_TEST, Y_TRAIN
from utils import helper


def create_dataset(X, y, time_step=TIME_STEP, space_window=SPACE_WINDOW):
    dataset = STDataset(X, y, time_step=time_step, space_window=space_window)
    return dataset


def create_loader(dataset, batch_size=BATCH_SIZE, shuffle=False):
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        worker_init_fn=helper.seed_worker,
        generator=g,
    )
    return dataloader


def tensor2loader(
    X,
    y,
    time_step=TIME_STEP,
    space_window=SPACE_WINDOW,
    batch_size=BATCH_SIZE,
    shuffle=False,
    random_seed=RANDOM_SEED,
):
    dataset = create_dataset(
        X, y, time_step=time_step, space_window=space_window
    )

    if shuffle:
        helper.fix_seed(random_seed)

    loader = create_loader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def load_data(
    X_train_fname=X_TRAIN,
    X_test_fname=X_TEST,
    y_train_fname=Y_TRAIN,
    y_test_fname=Y_TEST,
    time_step=TIME_STEP,
    space_window=SPACE_WINDOW,
    batch_size=BATCH_SIZE,
    random_seed=RANDOM_SEED,
):
    """
    return dataloader of train and test data

    Parameters
    ----------
    X_train_fname: str
    X_test_fname: str
    y_train_fname: str
    y_test_fname: str
    time_step: int
    space_window: Tuple[int]
    batch_size: int
    random_seed: int

    Returns
    -------
    train_loader: DataLoader
    test_loader: DataLoader
    """
    X_train = torch.load(X_train_fname)
    X_test = torch.load(X_test_fname)
    y_train = torch.load(y_train_fname)
    y_test = torch.load(y_test_fname)

    train_loader = tensor2loader(
        X_train,
        y_train,
        time_step=time_step,
        space_window=space_window,
        batch_size=batch_size,
        shuffle=True,
        random_seed=random_seed,
    )
    test_loader = tensor2loader(
        X_test,
        y_test,
        time_step=time_step,
        space_window=space_window,
        batch_size=batch_size,
        shuffle=False,
        random_seed=random_seed,
    )

    return train_loader, test_loader
