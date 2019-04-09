import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Env(object):
    """
    The true value q∗(a) of each of the ten actions was selected according to a normal distribution with mean zero and unit variance, and then the actual
    rewards were selected according to a mean q∗(a) unit variance normal distribution, as suggested by these gray
    distributions.
    """
    def __init__(self, k):
        self.q_a_real = np.random.normal(size=k)

    def show_distribution(self):
        q_list = []
        distribution_list = []
        for q in self.q_a_real:
            for d in np.random.normal(loc=q, size=1000):
                q_list.append(q)
                distribution_list.append(d)

        q_a_distribution = pd.DataFrame({"q_a_real": q_list, "distribution": distribution_list})
        sns.violinplot(x="q_a_real", y="distribution", data=q_a_distribution)
        plt.show()

    def get_reward(self, action_id):
        q = self.q_a_real[action_id]
        return np.random.normal(loc=q, scale=0.5)


class Agent(object):
    """methods: e-greedy method, ucb, gradient"""
    def __init__(self, k=10, init_value=0.0, epsilon=0.0, method="e-greedy"):
        self.value_function = np.array([init_value] * k, dtype=np.float32)
        self.k = k
        self.step = 1
        self.epsilon = epsilon
        self.action_selected_count = {}
        self.method = method
        self.action_count = np.zeros_like(self.value_function, dtype=np.int32)

    def guess(self):
        # exploration or exploitation
        if np.random.random() < self.epsilon:
            return np.random.choice(self.k)
        action = None
        if self.method == "e-greedy":
            action = np.argmax(self.value_function)
        elif self.method == "ucb":
            action = np.argmax(self.value_function + 2 * np.sqrt(np.log(self.step) / (self.action_count + 1e-5)))
        check = np.where(self.value_function == self.value_function[action])[0]
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)

    def update(self, action, reward, *args):
        self.action_count[action] += 1

        self.action_selected_count[action] = self.action_selected_count.get(action, 0) + 1
        action_value_before = self.value_function[action]
        self.value_function[action] = self.value_function[action] + (1 / self.step) * (reward - self.value_function[action])
        # print("action {} changed from {} to {}".format(action, action_value_before, self.value_function[action]))
        self.step += 1


def train(agent, env):
    reward_receive = []
    for i in range(1000):
        action = agent.guess()
        reward = env.get_reward(action)
        reward_receive.append(reward)
        agent.update(action, reward)
    return reward_receive


def plot_mean_reward(reward_receive, title):
    mean_averaged_reward_receive = []
    for i in range(len(reward_receive)):
        start = max(0, i - 50)
        end = min(len(reward_receive) - 1, i + 50)
        mean_averaged_reward_receive.append(np.mean(reward_receive[start: end]))
    plt.plot(mean_averaged_reward_receive)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    k = 10
    env = Env(k=k)

    # e-greedy
    agent = Agent(k=k, init_value=1.0, epsilon=0.1)
    reward_receive = train(agent, env)

    # plot_mean_reward(reward_receive, "e-greedy")

    print("q_a_real")
    print(env.q_a_real)
    print("value_function")
    print(agent.value_function)
    print(agent.action_selected_count)

    # ucb
    agent = Agent(k=k, init_value=1.0, epsilon=0.0, method="ucb")
    reward_receive = train(agent, env)

    # plot_mean_reward(reward_receive, "ucb")

    print("q_a_real")
    print(env.q_a_real)
    print("value_function")
    print(agent.value_function)
    print(agent.action_selected_count)