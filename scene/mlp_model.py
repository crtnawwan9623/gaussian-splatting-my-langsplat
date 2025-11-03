import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.mlp_utils import MlpUtilityNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class MlpModel:
    def __init__(self):
        self.mlp = MlpUtilityNetwork(input_size=3, output_size=4).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, x):
        return self.mlp(x)

    def train_setting(self, training_args):
        if training_args.include_feature:
            l = [
                {'params': list(self.mlp.parameters()),
                'lr': training_args.mlp_lr_init,
                "name": "mlp"}
            ]
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, weight_decay=1e-3)

            self.mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_lr_init,
                                                        lr_final=training_args.mlp_lr_final,
                                                        lr_delay_mult=training_args.mlp_lr_delay_mult,
                                                        max_steps=training_args.mlp_lr_max_steps)
        else:
            for param in self.mlp.parameters():
                param.requires_grad_(False)


    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "mlp/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.mlp.state_dict(), os.path.join(out_weights_path, 'mlp.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "mlp"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "mlp/iteration_{}/mlp.pth".format(loaded_iter))
        self.mlp.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mlp":
                lr = self.mlp_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
