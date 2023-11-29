from environments import DoNothingEnv, RunAwayEnv_v0, RunAwayEnv_v1, WalkerEnv_v0


def make(name="RunAway", env_conf=None):
    if name == "RunAway-v0":
        return RunAwayEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            max_distance=env_conf.max_distance,
            min_distance=env_conf.min_distance,
            verbose=env_conf.verbose,
        )
    elif name == "RunAway-v1":
        return RunAwayEnv_v1(
            max_episode_steps=env_conf.max_episode_steps,
            max_distance=env_conf.max_distance,
            min_distance=env_conf.min_distance,
            verbose=env_conf.verbose,
        )
    elif name == "Walker-v0":
        return WalkerEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            max_acc=env_conf.max_acceleration,
            reward_normalization_factor=env_conf.reward_normalization_factor,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
        )
    elif name == "DoNothing":
        return DoNothingEnv(
            max_episode_steps=env_conf.max_episode_steps,
            max_distance=env_conf.max_distance,
            min_distance=env_conf.min_distance,
        )
    else:
        print("Environment not found")
