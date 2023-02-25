import torch
import torch.nn as nn

from torchrl.modules import (
    AdditiveGaussianWrapper,
    MLP,
    ProbabilisticActor,
    SafeModule,
    ValueOperator,
)
from torchrl.modules.distributions import TanhDelta


def get_critic(in_keys=["observation"], num_cells=[256, 256], out_features=1, activation_class=nn.ReLU):
    """ Returns a critic network"""
    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": num_cells,
        "out_features": out_features,
        "activation_class": activation_class,
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )
    return qvalue

def get_deterministic_actor(action_spec, in_keys=["observation"], num_cells=[256, 256], activation_class=nn.ReLU):
    """ Returns a deterministic actor network like it is used in DDPG and TD3"""
     
    actor_net_kwargs = {
        "num_cells": num_cells,
        "out_features": action_spec.shape[-1],
        "activation_class": activation_class,
    }

    actor_net = MLP(**actor_net_kwargs)
        
    dist_class = TanhDelta
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
    }

    in_keys_actor = in_keys
    actor_module = SafeModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "param",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["param"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=False,
    )
    return actor

