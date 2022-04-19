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
    
    # Declare result container
    results = {
        "value_estimates": np.zeros((len(agents), bandit.num_arm)),
        "scores": np.zeros((num_iter, len(agents))),
        "optimals": np.zeros((num_iter, len(agents)))
    }

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
                results["scores"][step, index] += reward
                if b_optimal_action:
                    results["optimals"][step, index] += 1

            results["value_estimates"][index, :] += agent.value_estimates

    # Average results
    results["value_estimates"] /= num_experiment
    results["scores"] /= num_experiment
    results["optimals"] /= num_experiment

    return results 
