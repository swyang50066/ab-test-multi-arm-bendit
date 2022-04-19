from tqdm import tqdm

import numpy as np

    
def runner(
    bandit, 
    agents, 
    num_iter=1000, 
    num_experiment=500, 
    random_seed=931016
):
    # Set numpy random seed
    np.random.seed(random_seed)
    
    # Declare outputs
    scores = np.zeros((num_iter, len(agents)))
    optimals = np.zeros_like(scores)

    # Run simulation
    for n in range(num_experiment):
        print("Experiment: ", n)
        
        # Set up environment
        bandit.setup()

        for index, agent in enumerate(agents):
            # Reset agent
            agent.reset()

            for step in tqdm(range(num_iter), desc="iterations"):
                # Select action under policy
                action = agent.select_action()
                
                # Pull bandit 
                reward, b_optimal_action = bandit.pull_arm(action)
                
                # Update agent
                agent.observe(reward)

                # Store results
                scores[step, index] += reward
                if b_optimal_action:
                    optimals[step, index] += 1

    # Return outputs
    scores /= num_experiment
    optimals /= num_experiment

    return scores, optimals
