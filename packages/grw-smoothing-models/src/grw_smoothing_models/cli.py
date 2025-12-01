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


cli.add_command(kinetics_arrange)
cli.add_command(kinetics_create_clips)


if __name__ == "__main__":
    cli()
