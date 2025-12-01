import os.path
from configparser import ConfigParser
from typing import Tuple, List, Dict

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler

from grw_smoothing_models.backbones.movinets.a3b.a3b_backbone import MovinetBackbone
from grw_smoothing_models.data.kinetics_ds import KineticsDs
from grw_smoothing_models.model.action_recognition_model import ActionRecognitionModel
from grw_smoothing_models.model.attention import TransformerParams


def ddp_setup() -> Tuple[int, int]:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    if global_rank == 0:
        print(f'world_size={dist.get_world_size()}')
    return global_rank, local_rank


def cleanup():
    dist.destroy_process_group()


def main(model_name: str,
         config: ConfigParser,
         batch_size: int,
         fps: int,
         N: int,
         num_workers: int,
         split: str):
    assert split in ['test', 'val']
    try:
        global_rank, local_rank = ddp_setup()
    except Exception as e:
        cleanup()
        raise e
    try:
        seed = torch.tensor([42], dtype=torch.int).to(local_rank)
        torch.distributed.broadcast(tensor=seed, src=0)
        torch.manual_seed(seed.item())
        data_home, model_path = read_configurations(config, model_name)
        transform_test = transforms.Compose([
            lambda v: v.to(torch.float32) / 255,
            transforms.Resize((300, 300)),
            transforms.CenterCrop(256)])
        video_clips_pkl_name = split + '.pkl'
        ds = KineticsDs(root=data_home,
                        video_clips_pkl_name=video_clips_pkl_name,
                        frames_per_clip=N,
                        step_between_clips=N,
                        split=split,
                        transform=transform_test,
                        frame_rate=fps)
        n = len(ds)
        print(f'len(ds)={n}')
        sampler = DistributedSampler(ds)
        dataloader = DataLoader(ds,
                                batch_size=batch_size,
                                pin_memory=True,
                                sampler=sampler,
                                drop_last=True,
                                shuffle=False,
                                num_workers=num_workers)
        snapshot: Dict = torch.load(model_path, map_location='cpu', weights_only=False)
        transformer_params: TransformerParams = snapshot['transformer_params']
        model, num_samples = ActionRecognitionModel.load(model_path=model_path,
                                                         video_backbone=MovinetBackbone(embed_dim=transformer_params.embed_dim,max_drop_path_rate=0),
                                                         dropout=0,
                                                         stochastic_depth=0,
                                                         load_ema=False)
        ema_model, _ = ActionRecognitionModel.load(model_path=model_path,
                                                   video_backbone=MovinetBackbone(embed_dim=transformer_params.embed_dim,max_drop_path_rate=0),
                                                   dropout=0,
                                                   stochastic_depth=0,
                                                   load_ema=True)
        print(f'transformer_params {model.transformer_params}')
        validator = Validator(model=model,
                              ema_model=ema_model,
                              dataloader=dataloader,
                              batch_size=batch_size)
        validation_loss_ema, accuracy_ema, validation_loss, accuracy = validator.validate()
        if global_rank == 0:
            print(
                f"Validation Loss EMA: {validation_loss_ema:.3f}, Validation accuracy EMA: {accuracy_ema:.3f}")
            print(
                f"Validation Loss: {validation_loss:.3f}, Validation accuracy: {accuracy:.3f}")

    finally:
        cleanup()


class Validator:
    def __init__(self,
                 model: ActionRecognitionModel,
                 ema_model: ActionRecognitionModel,
                 dataloader: DataLoader,
                 batch_size: int):
        self.batch_size: int = batch_size
        self.dataloader: DataLoader = dataloader
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = dist.get_world_size()
        self.local_rank: int = local_rank
        self.global_rank: int = global_rank
        self.world_size: int = world_size

        model.to(local_rank)
        self.model: ActionRecognitionModel = model
        ema_model.to(local_rank)
        self.ema_model: ActionRecognitionModel = ema_model

    def set_eval(self):
        self.model.eval()
        self.ema_model.eval()

    def validate(self) -> Tuple[float, float, float, float]:
        self.set_eval()
        ce_loss_fn_sum = torch.nn.CrossEntropyLoss(reduction='sum')
        loss_aggregate_ema = torch.tensor([0.0], device=self.local_rank, dtype=torch.float32)
        loss_aggregate = torch.tensor([0.0], device=self.local_rank, dtype=torch.float32)

        y_tensor_list_for_node: List[Tensor] = []
        y_hat_ema_tensor_list_for_node: List[Tensor] = []
        y_hat_tensor_list_for_node: List[Tensor] = []
        N = 0
        with torch.no_grad():

            for batch_idx, (X, Y) in enumerate(self.dataloader):
                X = X.to(self.local_rank, non_blocking=True)
                Y = Y.to(self.local_rank, non_blocking=True)
                y_tensor_list_for_node.append(Y)

                logits_ema, _ = self.ema_model(X)
                _, y_hat_ema = torch.max(logits_ema, 1)
                y_hat_ema_tensor_list_for_node.append(y_hat_ema)

                logits, _ = self.model(X)
                _, y_hat = torch.max(logits, 1)
                y_hat_tensor_list_for_node.append(y_hat)

                loss_ema = ce_loss_fn_sum(logits_ema, Y)
                loss_aggregate_ema += loss_ema

                loss = ce_loss_fn_sum(logits, Y)
                loss_aggregate += loss

                N += 1
                if batch_idx % 10 == 0 and self.global_rank == 0:
                    print(f" val loss_ema: {loss_aggregate_ema.item() / (N * self.batch_size):>7f}  ")
                    print(f" val loss: {loss_aggregate.item() / (N * self.batch_size):>7f}  ")
                    print(f"[{(batch_idx + 1) * self.batch_size * self.world_size:>5d}/"
                          f"{len(self.dataloader) * self.batch_size * self.world_size:>5d}]")

            dist.all_reduce(loss_aggregate_ema, op=dist.ReduceOp.AVG)
            validation_loss_ema = loss_aggregate_ema / (N * self.batch_size)

            dist.all_reduce(loss_aggregate, op=dist.ReduceOp.AVG)
            validation_loss = loss_aggregate / (N * self.batch_size)

            y_: Tensor = torch.cat(y_tensor_list_for_node)

            y_hat_ema_: Tensor = torch.cat(y_hat_ema_tensor_list_for_node)
            accuracy_ema = accuracy_score(y_.cpu(), y_hat_ema_.cpu())
            accuracy_ema = torch.tensor([accuracy_ema], dtype=torch.float32, device=self.local_rank)
            dist.all_reduce(accuracy_ema, op=dist.ReduceOp.AVG)

            y_hat_: Tensor = torch.cat(y_hat_tensor_list_for_node)
            accuracy = accuracy_score(y_.cpu(), y_hat_.cpu())
            accuracy = torch.tensor([accuracy], dtype=torch.float32, device=self.local_rank)
            dist.all_reduce(accuracy, op=dist.ReduceOp.AVG)

        return validation_loss_ema.item(), accuracy_ema.item(), validation_loss.item(), accuracy.item()


def read_configurations(config, model_name):
    data_home = config.get('kinetics', 'data_home')
    models_home = config.get('global', 'models_home')
    model_path = str(os.path.join(models_home, model_name))
    return data_home, model_path
