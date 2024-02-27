from environments import (
    BalanceEnv_v0,
    DoNothingEnv,
    RoboArmEnv_v0,
    RoboArmMixedEnv_v0,
    RunAwayEnv_v0,
    RunAwayEnv_v1,
    SpinningEnv_v1,
    WalkerEnv_v0,
    WalkerEnv_v1,
    WalkerEnv_v2,
    WalkerMixedEnv_v0,
)


def make(name="RunAway", env_conf=None):
    if name == "runaway-v0":
        return RunAwayEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            min_distance=env_conf.min_distance,
            verbose=env_conf.verbose,
        )
    elif name == "runaway-v1":
        return RunAwayEnv_v1(
            max_episode_steps=env_conf.max_episode_steps,
            min_distance=env_conf.min_distance,
            verbose=env_conf.verbose,
        )
    elif name == "spinning-v1":
        return SpinningEnv_v1(
            max_episode_steps=env_conf.max_episode_steps,
            sleep_time=env_conf.sleep_time,
            verbose=env_conf.verbose,
        )
    elif name == "balance-v0":
        return BalanceEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            sleep_time=env_conf.sleep_time,
            verbose=env_conf.verbose,
        )
    elif name == "walker-v0":
        return WalkerEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
        )
    elif name == "walker-v1":
        return WalkerEnv_v1(
            max_episode_steps=env_conf.max_episode_steps,
            max_acc=env_conf.max_acceleration,
            reward_normalization_factor=env_conf.reward_normalization_factor,
            reward_clip=env_conf.reward_clip,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
        )
    elif name == "walker-v2":
        return WalkerEnv_v2(
            max_episode_steps=env_conf.max_episode_steps,
            max_acc=env_conf.max_acceleration,
            max_distance=env_conf.max_distance,
            reward_normalization_factor=env_conf.reward_normalization_factor,
            reward_clip_acc=env_conf.reward_clip_acc,
            use_acceleration_reward=env_conf.use_acceleration_reward,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
        )
    elif name == "walker_mixed-v0":
        return WalkerMixedEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            sleep_time=env_conf.sleep_time,
            verbose=env_conf.verbose,
            camera_id=env_conf.camera_id,
            image_path=env_conf.image_path,
            tracker_type=env_conf.tracker_type,
            auto_obj_detection=env_conf.auto_obj_detection,
        )
    elif name == "roboarm-v0":
        return RoboArmEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
            reward_signal=env_conf.reward_signal,
        )
    elif name == "roboarm_mixed-v0":
        return RoboArmMixedEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            sleep_time=env_conf.sleep_time,
            verbose=env_conf.verbose,
            reward_signal=env_conf.reward_signal,
            camera_id=env_conf.camera_id,
            goal_radius=env_conf.goal_radius,
        )
    elif name == "DoNothing":
        return DoNothingEnv(
            max_episode_steps=env_conf.max_episode_steps,
            max_distance=env_conf.max_distance,
            min_distance=env_conf.min_distance,
        )
    else:
        print("Environment not found")
