import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.modules import (
    AdditiveGaussianWrapper,
    MLP,
    ProbabilisticActor,
    SafeSequential,
    TanhModule,
    SafeModule,
    ValueOperator,
)
from torchrl.modules.distributions import TanhDelta, TanhNormal

def get_normalization(normalization):
    if normalization == "None":
        return None
    elif normalization == "LayerNorm":
        return nn.LayerNorm
    elif normalization == "BatchNorm":
        return nn.BatchNorm1d
    else:
        raise NotImplementedError(f"Normalization {normalization} not implemented")

def get_critic(
    in_keys=["observation"],
    num_cells=[256, 256],
    out_features=1,
    activation_class=nn.ReLU,
    normalization="None",
    dropout=0.0,
):
    """Returns a critic network"""
    normalization = get_normalization(normalization)
    qvalue_net = MLP(
        num_cells=num_cells,
        out_features=out_features,
        activation_class=activation_class,
        norm_class=normalization,
        norm_kwargs={"normalized_shape": num_cells[-1]} if normalization else None,
        dropout=dropout,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )
    return qvalue


def get_deterministic_actor(
    action_spec,
    in_keys=["observation"],
    num_cells=[256, 256],
    activation_class=nn.ReLU,
    normalization="None",
    dropout=0.0,
):
    """Returns a deterministic actor network like it is used in DDPG and TD3"""
    normalization = get_normalization(normalization)
    actor_net = MLP(
        num_cells=num_cells,
        out_features=action_spec.shape[-1],
        activation_class=activation_class,
        norm_class=normalization,
        norm_kwargs={"normalized_shape": num_cells[-1]} if normalization else None,
        dropout=dropout,
    )

    in_keys_actor = in_keys
    actor_module = SafeModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "param",
        ],
    )
    actor = SafeSequential(
        actor_module,
        TanhModule(
            in_keys=["param"],
            out_keys=["action"],
            spec=action_spec,
        ),
    )

    return actor


def get_stochastic_actor(
    action_spec,
    in_keys=["observation"],
    num_cells=[256, 256],
    normalization="None",
    dropout=0.0,
    activation_class=nn.ReLU,
):
    """Returns a stochastic actor network like it is used in SAC"""
    normalization = get_normalization(normalization)
    actor_net = MLP(
        num_cells=num_cells,
        out_features=2 * action_spec.shape[-1],
        activation_class=activation_class,
        norm_class=normalization,
        norm_kwargs={"normalized_shape": num_cells[-1]} if normalization else None,
        dropout=dropout,
    )

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": False,
    }
    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{1.0}",
        scale_lb=0.1,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)
    actor_module = SafeModule(
        actor_net,
        in_keys=in_keys,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=False,
    )
    return actor
