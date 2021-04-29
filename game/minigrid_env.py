from gym_minigrid.wrappers import *
import gym
import cv2

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale

        low_val = env.observation_space['image'].low.min()
        high_val = env.observation_space['image'].high.max()
        w, h, c = env.observation_space['image'].shape


        if self.grayscale:
            self.observation_space = spaces.Box(low=low_val, high=high_val,
                shape=(1, w, h), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=low_val, high=high_val,
                shape=(3, w, h), dtype=np.uint8)

    def observation(self, frame):
        frame = frame['image']
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.width != None:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        frame = frame.transpose([2, 0, 1])
        return frame


class MinigridEnv(gym.Wrapper):
    def __init__(self, env):
        env = RGBImgObsWrapper(env)
        self.env = WarpFrame(env, width=None, height=None)
        super(MinigridEnv, self).__init__(self.env)
        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return
         ) = get_gym_env_info(self.env, True)




def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
            gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
            assert isinstance(env, gym.Env)
            env_name = env.unwrapped.spec.id
            state_dim = list(env.observation_space.shape)
            target_reward = getattr(env, 'target_reward', None)
            target_reward_default = getattr(env.spec, 'reward_threshold', None)
            if target_reward is None:
                target_reward = target_reward_default
            if target_reward is None:
                target_reward = 2 ** 16
            max_step = getattr(env, 'max_step', None)
            max_step_default = getattr(env, '_max_episode_steps', None)
            if max_step is None:
                max_step = max_step_default
            if max_step is None:
                max_step = 2 ** 10
            if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
            if if_discrete:  # make sure it is discrete action space
                action_dim = env.action_space.n
                action_max = int(1)
            elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
                action_dim = env.action_space.shape[0]
                action_max = float(env.action_space.high[0])
            else:
                raise RuntimeError(
                    '| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')
            print(f"\n| env_name:  {env_name}, action space if_discrete: {if_discrete}"
                  f"\n| state_dim: {state_dim}, action_dim: {action_dim}, action_max: {action_max}"
                  f"\n| max_step:  {max_step}, target_reward: {target_reward}") if if_print else None
            return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_reward




if __name__ == "__main__":
    env = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
    env = MinigridEnv(env)
    obs = env.reset()
    print(obs.shape)
    while True:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

