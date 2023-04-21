import gym

env = gym.make('CartPole-v1', render_mode='rgb_array') # human"

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1 # when control pole leans left, accelerate left, otherwise right

totals = []
for episode in range(500):
    episode_rewards = 0
    obs, info = env.reset()
    print(obs, info)

    for step in range(1000): # 1000 steps at most
        action = basic_policy(obs)
        obs, reward, done, trunc, info = env.step(action)
        print(obs, reward, done, trunc, info)
        episode_rewards += reward
        if done:
            break
        print('episode rewards: ', episode_rewards)

    totals.append(episode_rewards)
print(totals)
