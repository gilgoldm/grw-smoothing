import os

import click


@click.group()
def cli():
    """
    My Awesome CLI Tool
    """
    pass




@click.command()
def kinetics_arrange():
    from config.grw_smoothing_models_config import GrwSmoothingModelsConfig
    from grw_smoothing_models.data.arrange import KineticsDataSet
    config = GrwSmoothingModelsConfig('../../config.ini').config
    data_home = config.get('kinetics', 'data_home')
    kinetics_data_set = KineticsDataSet(data_home)
    kinetics_data_set.arrange_as_pytorchvision_kinetics_dataset()


@click.command()
@click.option('--num_workers', default=12, type=int, help='num_workers')
def kinetics_create_clips(num_workers: int):
    from config.grw_smoothing_models_config import GrwSmoothingModelsConfig
    from data.kinetics_ds import KineticsDs
    config = GrwSmoothingModelsConfig('../../config.ini').config
    data_home = config.get('kinetics', 'data_home')

    KineticsDs.calculate_metadata(pkl_name='train.pkl',
                       root=data_home,
                       split='train',
                       step_between_clips=5,
                       num_workers=num_workers,
                       frames_per_clip=45,
                       frame_rate=5,
                       extensions=("avi", "mp4"))

    KineticsDs.calculate_metadata(pkl_name='val.pkl',
                       root=data_home,
                       split='val',
                       step_between_clips=5,
                       num_workers=num_workers,
                       frames_per_clip=45,
                       frame_rate=5,
                       extensions=("avi", "mp4"))

    KineticsDs.calculate_metadata(pkl_name='test.pkl',
                       root=data_home,
                       split='test',
                       step_between_clips=5,
                       num_workers=num_workers,
                       frames_per_clip=45,
                       frame_rate=5,
                       extensions=("avi", "mp4"))


@click.command()
@click.option('--model_name', type=str, help='Name of the model')
@click.option('--split', type=str, default='test', help='test / val')
@click.option('--batch_size', type=int, help='Batch size')
@click.option('--num_workers', type=int, required=False, default=4, help='Number of loader workers')
@click.option('--fps', type=int, help='Frames per second')
@click.option('--N', 'N', type=int, help='Clip frames')
@click.option('--backbone', type=str, help='Spatial backbone')
def kinetics_test(model_name: str,
                  batch_size: int,
                  num_workers: int,
                  split: str,
                  fps: int,
                  N: int,
                  backbone: str):
    from typing import Dict
    import torch

    from grw_smoothing_models.config.grw_smoothing_models_config import GrwSmoothingModelsConfig
    from grw_smoothing_models.model.attention import TransformerParams

    config = GrwSmoothingModelsConfig('../../config.ini').config
    models_home = config.get('global', 'models_home')
    model_path = str(os.path.join(models_home, model_name))

    snapshot: Dict = torch.load(model_path, map_location='cpu', weights_only=False)
    transformer_params: TransformerParams = snapshot['transformer_params']

    if backbone == 'movineta0s':
        from grw_smoothing_models.backbones.movinets.a0s.a0s_backbone import MovinetBackbone
        backbone_ = MovinetBackbone(embed_dim=transformer_params.embed_dim, max_drop_path_rate=0)
        ema_backbone = MovinetBackbone(embed_dim=transformer_params.embed_dim, max_drop_path_rate=0)
    elif backbone == 'movineta1s':
        from grw_smoothing_models.backbones.movinets.a1s.a1s_backbone import MovinetBackbone
        backbone_ = MovinetBackbone(embed_dim=transformer_params.embed_dim, max_drop_path_rate=0)
        ema_backbone = MovinetBackbone(embed_dim=transformer_params.embed_dim, max_drop_path_rate=0)
    elif backbone == 'movineta2s':
        from grw_smoothing_models.backbones.movinets.a2s.a2s_backbone import MovinetBackbone
        backbone_ = MovinetBackbone(embed_dim=transformer_params.embed_dim, max_drop_path_rate=0)
        ema_backbone = MovinetBackbone(embed_dim=transformer_params.embed_dim, max_drop_path_rate=0)
    elif backbone == 'movineta3b':
        from grw_smoothing_models.backbones.movinets.a3b.a3b_backbone import MovinetBackbone
        backbone_ = MovinetBackbone(embed_dim=transformer_params.embed_dim, max_drop_path_rate=0)
        ema_backbone = MovinetBackbone(embed_dim=transformer_params.embed_dim, max_drop_path_rate=0)
    else:
        raise Exception(f'backbone={backbone} unknown')

    from grw_smoothing_models.backbones.test import main as test
    test(model_name=model_name,
         config=config,
         split=split,
         batch_size=batch_size,
         num_workers=num_workers,
         N=N,
         fps=fps,
         backbone=backbone_,
         ema_backbone=ema_backbone)

cli.add_command(kinetics_arrange)
cli.add_command(kinetics_create_clips)
cli.add_command(kinetics_test)


if __name__ == "__main__":
    cli()
