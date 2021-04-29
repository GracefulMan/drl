import cv2
import torch
import time
import numpy as np
def test_d3qn():

    from elegantrl.agent import AgentD3QN
    from elegantrl.atari_env import AtariGameEnv
    import gym
    agent = AgentD3QN()
    env = AtariGameEnv(env=gym.make('Breakout-v0'))
    net_dim = 2 ** 9  # change a default hyper-parameters
    agent.init(net_dim, env.state_dim, env.action_dim)
    agent.save_load_model(cwd='AgentD3QN/Breakout-v0_0', if_save=False)
    state = env.reset()
    episode_return = 0
    step = 0
    for i in range(10 ** 5):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        cv2.imwrite(f'img/{i}.png', np.array(next_state[-1].squeeze()*255, dtype=np.int))
        episode_return += reward
        step += 1
        if done:
            print(f'{i:>6}, {step:6.0f}, {episode_return:8.3f}, {reward:8.3f}')
            state = env.reset()
            episode_return = 0
            step = 0
        state = next_state
        #env.render()
        time.sleep(0.1)



if __name__ == '__main__':
    test_d3qn()