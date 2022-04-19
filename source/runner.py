import numpy as np

    
def runner(bandit, agents, num_iter=100, num_experiment=1):
    
    scores = np.zeros((num_iter, len(agents)))
    optimal = np.zeros_like(scores)

    for _ in range(num_experiment):
        for step in range(num_iter):
            for index, agent in enumerate(agents):
                action = agent.select_action()
                
                reward, is_optimal = bandit.pull_arm(action)
                
                agent.observe(reward)

                scores[step, index] += reward
                if is_optimal:
                    optimal[step, index] += 1

                print(
                    "step", step, 
                    "index", index,
                    "reward",  reward
                )

    return scores / num_experiment, optimal / num_experiment
