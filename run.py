import carla
from min_carla_env.env import CarlaEnv, CONFIG


if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    for _ in range(5):
        try:
            env = CarlaEnv(client, CONFIG, debug=True)
            old_obs = env.reset()
            done = False
            t = 0
            total_reward = 0.0
            while not done:
                t += 1
                obs, reward, done, info = env.step(0)  # Go Forward
                total_reward += reward
                print("step#:", t, "reward:", round(reward, 4),
                      "total_reward:", round(total_reward, 4), "done:", done)
        finally:
            env.mw.clean_world()
