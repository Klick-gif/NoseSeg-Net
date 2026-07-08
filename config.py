import argparse
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    data_root: str = "dataset"
    cache_root: str = "cache"
    use_cache: bool = False
    image_size: int = 512
    skip_empty: bool = False
    train_ratio: float = 0.8

    batch_size: int = 4
    shuffle: bool = True
    num_workers: int = 6
    pin_memory: bool = True

    in_channels: int = 1
    num_classes: int = 2
    epochs: int = 20
    lr: float = 1e-4
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    model_name: str = "unet"
    log_root: str = "runs"
    result_root: str = "results"
    save_name: str = "best_unet_model.pth"


def get_config():
    parser = argparse.ArgumentParser(description="Train nose segmentation model")

    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--data_root", type=str, default=Config.data_root)
    parser.add_argument("--cache_root", type=str, default=Config.cache_root)
    parser.add_argument("--use_cache", action="store_true", default=Config.use_cache)
    parser.add_argument("--image_size", type=int, default=Config.image_size)
    parser.add_argument("--skip_empty", action="store_true", default=Config.skip_empty)
    parser.add_argument("--train_ratio", type=float, default=Config.train_ratio)

    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--num_workers", type=int, default=Config.num_workers)
    parser.add_argument("--no_pin_memory", action="store_true")

    parser.add_argument("--in_channels", type=int, default=Config.in_channels)
    parser.add_argument("--num_classes", type=int, default=Config.num_classes)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--scheduler_patience", type=int, default=Config.scheduler_patience)
    parser.add_argument("--scheduler_factor", type=float, default=Config.scheduler_factor)

    parser.add_argument("--model_name", type=str, default=Config.model_name)
    parser.add_argument("--log_root", type=str, default=Config.log_root)
    parser.add_argument("--result_root", type=str, default=Config.result_root)
    parser.add_argument("--save_name", type=str, default=Config.save_name)

    args = parser.parse_args()
    return Config(
        seed=args.seed,
        data_root=args.data_root,
        cache_root=args.cache_root,
        use_cache=args.use_cache,
        image_size=args.image_size,
        skip_empty=args.skip_empty,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        epochs=args.epochs,
        lr=args.lr,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        model_name=args.model_name,
        log_root=args.log_root,
        result_root=args.result_root,
        save_name=args.save_name,
    )
