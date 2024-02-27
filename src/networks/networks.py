import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.modules import (
    AdditiveGaussianWrapper,
    ConvNet,
    MLP,
    ProbabilisticActor,
    SafeModule,
    SafeSequential,
    TanhModule,
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


def get_critic(observation_keys, agent_config):
    if "vec_observation" in observation_keys and not "image_observation" in observation_keys:
        return get_vec_critic(
            in_keys=observation_keys,
            num_cells=[agent_config.num_cells, agent_config.num_cells],
            out_features=1,
            activation_class=nn.ReLU,
            normalization=agent_config.normalization,
            dropout=agent_config.dropout,
        )
    elif (
        "image_observation" in observation_keys
        and "vec_observation" in observation_keys
    ):
        return get_mixed_critic(
            vec_in_keys="vec_observation",
            img_in_keys="image_observation",
            num_cells=[agent_config.num_cells, agent_config.num_cells],
            out_features=1,
            activation_class=nn.ReLU,
            normalization=agent_config.normalization,
            dropout=agent_config.dropout,
        )
    else:
        raise NotImplementedError("Critic for this observation space not implemented")


def get_vec_critic(
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


def get_mixed_critic(
    vec_in_keys,
    img_in_keys,
    num_cells=[256, 256],
    out_features=1,
    activation_class=nn.ReLU,
    normalization="None",
    dropout=0.0,
):
    normalization = get_normalization(normalization)
    # image encoder
    cnn = ConvNet(
        activation_class=activation_class,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    )
    cnn_output = cnn(torch.ones((3, 64, 64)))
    mlp = MLP(
        in_features=cnn_output.shape[-1],
        activation_class=activation_class,
        out_features=128,
        num_cells=[256],
    )
    image_encoder = SafeModule(
        torch.nn.Sequential(cnn, mlp),
        in_keys=[img_in_keys],
        out_keys=["image_embedding"],
    )

    # vector_obs encoder
    mlp = MLP(
        activation_class=activation_class,
        out_features=32,
        num_cells=[128],
    )
    vector_obs_encoder = SafeModule(
        mlp, in_keys=[vec_in_keys], out_keys=["vec_obs_embedding"]
    )

    # output head
    mlp = MLP(
        activation_class=torch.nn.ReLU,
        out_features=out_features,
        num_cells=num_cells,
        norm_class=normalization,
        norm_kwargs={"normalized_shape": num_cells[-1]} if normalization else None,
        dropout=dropout,
    )
    v_head = ValueOperator(mlp, ["image_embedding", "vec_obs_embedding", "action"])
    # model
    return SafeSequential(image_encoder, vector_obs_encoder, v_head)


def get_deterministic_actor(observation_keys, action_spec, agent_config):
    if len(observation_keys) == 1 and "vec_observation" in observation_keys:
        return get_vec_deterministic_actor(
            in_keys=observation_keys,
            num_cells=[agent_config.num_cells, agent_config.num_cells],
            activation_class=nn.ReLU,
        )

    elif (
        "image_observation" in observation_keys
        and "vec_observation" in observation_keys
    ):
        return get_mixed_deterministic_actor(
            vec_in_keys="vec_observation",
            img_in_keys="image_observation",
            action_spec=action_spec,
            num_cells=[agent_config.num_cells, agent_config.num_cells],
            activation_class=nn.ReLU,
        )
    else:
        raise NotImplementedError("Actor for this observation space not implemented")


def get_vec_deterministic_actor(
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


def get_mixed_deterministic_actor(
    vec_in_keys,
    img_in_keys,
    action_spec,
    num_cells=[256, 256],
    activation_class=nn.ReLU,
    normalization="None",
    dropout=0.0,
):
    normalization = get_normalization(normalization)
    # image encoder
    cnn = ConvNet(
        activation_class=activation_class,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    )
    cnn_output = cnn(torch.ones((3, 64, 64)))
    mlp = MLP(
        in_features=cnn_output.shape[-1],
        activation_class=activation_class,
        out_features=128,
        num_cells=[256],
    )
    image_encoder = SafeModule(
        torch.nn.Sequential(cnn, mlp),
        in_keys=[img_in_keys],
        out_keys=["image_embedding"],
    )

    # vector_obs encoder
    mlp = MLP(
        activation_class=activation_class,
        out_features=32,
        num_cells=[128],
    )
    vector_obs_encoder = SafeModule(
        mlp, in_keys=[vec_in_keys], out_keys=["vector_obs_embedding"]
    )

    # output head
    mlp = MLP(
        activation_class=torch.nn.ReLU,
        out_features=action_spec.shape[-1],
        num_cells=num_cells,
        norm_class=normalization,
        norm_kwargs={"normalized_shape": num_cells[-1]} if normalization else None,
        dropout=dropout,
    )
    combined = SafeModule(
        mlp, ["image_embedding", "vector_obs_embedding"], out_keys=["params"]
    )
    out_module = TanhModule(
        in_keys=["params"],
        out_keys=["action"],
        spec=action_spec,
    )
    return SafeSequential(
        image_encoder,
        vector_obs_encoder,
        combined,
        out_module,
    )


def get_stochastic_actor(observation_keys, action_spec, agent_config):
    if "vec_observation" in observation_keys and not "image_observation" in observation_keys:
        return get_vec_stochastic_actor(
            action_spec,
            in_keys=observation_keys,
            num_cells=[agent_config.num_cells, agent_config.num_cells],
            normalization=agent_config.normalization,
            dropout=agent_config.dropout,
            activation_class=nn.ReLU,
        )
    elif (
        "image_observation" in observation_keys
        and "vec_observation" in observation_keys
    ):
        return get_mixed_stochastic_actor(
            action_spec,
            vec_in_keys="vec_observation",
            img_in_keys="image_observation",
            num_cells=[agent_config.num_cells, agent_config.num_cells],
            normalization=agent_config.normalization,
            dropout=agent_config.dropout,
            activation_class=nn.ReLU,
        )
    else:
        raise NotImplementedError("Actor for this observation space not implemented")


def get_vec_stochastic_actor(
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
    actor_module = SafeModule(
        actor_net,
        in_keys=in_keys,
        out_keys=[
            "params"
        ],
    )
    extractor_module = SafeModule(
        actor_extractor,
        in_keys=["params"],
        out_keys=["loc", "scale"],)

    actor_module = SafeSequential(actor_module, extractor_module)
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


def get_mixed_stochastic_actor(
    action_spec,
    vec_in_keys,
    img_in_keys,
    num_cells=[256, 256],
    normalization="None",
    dropout=0.0,
    activation_class=nn.ReLU,
):

    normalization = get_normalization(normalization)
    # image encoder
    cnn = ConvNet(
        activation_class=activation_class,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    )
    cnn_output = cnn(torch.ones((3, 64, 64)))
    mlp = MLP(
        in_features=cnn_output.shape[-1],
        activation_class=activation_class,
        out_features=128,
        num_cells=[256],
    )
    image_encoder = SafeModule(
        torch.nn.Sequential(cnn, mlp),
        in_keys=[img_in_keys],
        out_keys=["image_embedding"],
    )

    # vector_obs encoder
    mlp = MLP(
        activation_class=activation_class,
        out_features=32,
        num_cells=[128],
    )
    vector_obs_encoder = SafeModule(
        mlp, in_keys=[vec_in_keys], out_keys=["vector_obs_embedding"]
    )

    # output head
    mlp = MLP(
        activation_class=torch.nn.ReLU,
        out_features=2 * action_spec.shape[-1],
        num_cells=num_cells,
        norm_class=normalization,
        norm_kwargs={"normalized_shape": num_cells[-1]} if normalization else None,
        dropout=dropout,
    )
    actor_module = SafeModule(
        mlp,
        in_keys=["image_embedding", "vector_obs_embedding"],
        out_keys=["params"],
    )
    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{1.0}",
        scale_lb=0.1,
    )

    extractor_module = SafeModule(
        actor_extractor,
        in_keys=["params"],
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor_net_combined = SafeSequential(
        image_encoder, vector_obs_encoder, actor_module, extractor_module
    )

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": False,
    }
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        module=actor_net_combined,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=False,
    )
    return actor