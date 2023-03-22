from environments import RunAwayEnv, DoNothingEnv, BalanceEnv

def make(name="RunAway", max_episode_steps=10,):
    if name == "RunAway":
        return RunAwayEnv(max_episode_steps=max_episode_steps,
                          max_distance=1000.,
                          min_distance=40.)
    elif name == "DoNothing":
        return DoNothingEnv(max_episode_steps=max_episode_steps,
                            max_distance=1000.,
                            min_distance=40.)
    elif name == "Balance":
        return BalanceEnv(max_episode_steps=max_episode_steps)
    else:
        print("Environment not found")