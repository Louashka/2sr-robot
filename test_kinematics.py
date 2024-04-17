from View import agent_plotlib as ap
from Model import global_var as gv, agent
import numpy as np


if __name__ == "__main__":
    robot2sr = agent.Robot(1, 0, 0, np.pi / 2, 15, -22)

    v = np.array([0, 0, 0.5]).T

    total_t = 10
    t = 0
    dt = 0.05

    config_array = []

    while t < total_t:

        robot2sr.update(v, dt)
        config_array.append(robot2sr.config)

        t += dt

    ap.animateAgent(robot2sr.id, config_array, dt)

