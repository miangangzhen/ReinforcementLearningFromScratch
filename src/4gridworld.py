import numpy as np
np.random.seed(4321)

WORLD_SIZE = 4
ACTIONS = [
    [1, 0], # down
    [-1, 0], # up
    [0, 1], # right
    [0, -1] # left
    ]
ACTION_PROB = 0.25


def init_sandbox():
    """
    初始化沙盒
    :return:
    """
    sandbox = np.zeros([WORLD_SIZE, WORLD_SIZE])
    position = (np.random.randint(0, WORLD_SIZE), np.random.randint(0, WORLD_SIZE))
    sandbox[position] += 1
    return sandbox.reshape([WORLD_SIZE, WORLD_SIZE]), position


def is_valide(position):
    """
    判断是否在有效位置
    :param position:
    :return:
    """
    x, y = position
    return x >= 0 and x < WORLD_SIZE and y >= 0 and y < WORLD_SIZE


def is_terminal(position):
    """
    在左上角或右下角时，游戏结束
    :param position:
    :return:
    """
    x, y = position
    return (x == 0 and y ==0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


def random_walk(sandbox, position):
    """
    随机游走，上下左右四个方向随机走一步
    :param sandbox:
    :param position:
    :return:
    """
    while(True):
        action_id = np.random.randint(0, len(ACTIONS))
        position_new = (position[0] + ACTIONS[action_id][0], position[1] + ACTIONS[action_id][1])
        if is_valide(position_new):
            break

    sandbox[position] -= 1
    sandbox[position_new] += 1

    return sandbox, position_new


def step(state, action):
    state = np.array(state)
    next_state = (state + action).tolist()
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state.tolist()

    reward = -1
    return next_state, reward

def compute_state_value(in_place = False):
    new_state_value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    state_value = new_state_value.copy()
    iteration = 1
    while True:
        src = new_state_value if in_place else state_value
        # 遍历每个位置
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                if is_terminal((i, j)):
                    continue
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step((i, j), action)
                    value += ACTION_PROB * (reward + src[next_i, next_j])
                new_state_value[i, j] = value

        if np.sum(np.abs(new_state_value - state_value)) < 1e-4:
            state_value = new_state_value.copy()
            break
        state_value = new_state_value.copy()
        iteration += 1

    return state_value, iteration

if __name__ == "__main__":
    sandbox, position = init_sandbox()
    print(sandbox)
    print(position)

    sandbox, position = random_walk(sandbox, position)
    print(sandbox)
    print(position)

    value, sync_iteration = compute_state_value()
    print(sync_iteration)
    print(value)