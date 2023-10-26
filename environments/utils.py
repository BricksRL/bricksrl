from environments import DoNothingEnv, RunAwayEnv


def make(name="RunAway", env_conf=None):
    if name == "RunAway":
        return RunAwayEnv(
            max_episode_steps=env_conf.max_episode_steps,
            max_distance=env_conf.max_distance,
            min_distance=env_conf.min_distance,
        )
    elif name == "DoNothing":
        return DoNothingEnv(
            max_episode_steps=env_conf.max_episode_steps,
            max_distance=env_conf.max_distance,
            min_distance=env_conf.min_distance,
        )
    else:
        print("Environment not found")
