import os
import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

def agent_wapper(config, agent):
    def fn(obs):
        with torch.no_grad():
            obs = obs.astype(np.float32)
            assert len(obs.shape) == 2
            obs = torch.from_numpy(obs).to(config.device)
            action = agent(obs)
        return action.cpu().numpy()
    return fn

def fit_dataset(config, agent, dataset, n_epochs):
    optimizer = optim.Adam(agent.parameters(), lr=config.lr, weight_decay=config.L2)
    loss_fn = nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    step = 0
    best_reward = None
    loss_his = []
    
    for k in range(n_epochs):
        for batch in dataloader:
            obs, gold_actions = batch
            pred_actions = agent(obs)
            loss = loss_fn(pred_actions, gold_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_his.append(loss.item())

            if step % config.eval_steps == 0:
                avrg_mean, avrg_std = Eval(config, agent_wapper(config, agent))
                avrg_loss = np.mean(loss_his)
                loss_his = []
                print('[epoch {}  step {}] loss: {:.4f}  r_mean: {:.2f}  r_std: {:.2f}'.format(
                    k + 1, step, avrg_loss, avrg_mean, avrg_std))

                avrg_reward = avrg_mean - avrg_std
                if best_reward is None or best_reward < avrg_reward:
                    best_reward = avrg_reward
                    save_model(config, agent, config.model_save_path)
                
            step += 1
    
    load_model(config, agent, config.model_save_path)

def BehavioralCloning(config, agent, expert):

    # get expert demonstration
    expert_obs, expert_actions, *_ = run_agent(config, expert, config.n_expert_rollouts)
    expert_obs = torch.from_numpy(expert_obs).to(config.device)
    expert_actions = torch.from_numpy(expert_actions).to(config.device)
    dataset = TensorDataset(expert_obs, expert_actions)

    # training agent
    fit_dataset(config, agent, dataset, config.epochs)

    return agent_wapper(config, agent)

def DAgger(config, agent, expert):
    # get expert demonstration
    expert_obs, expert_actions, *_ = run_agent(config, expert, config.n_expert_rollouts)
    expert_obs = torch.from_numpy(expert_obs).to(config.device)
    expert_actions = torch.from_numpy(expert_actions).to(config.device)
    dataset = TensorDataset(expert_obs, expert_actions)

    for k in range(config.n_dagger_iter):
        # training agent
        fit_dataset(config, agent, dataset, config.epochs)
        
        # run agent to get new on-policy observations
        new_obs, *_ = run_agent(config, agent_wapper(config, agent), config.n_dagger_rollouts)
        expert_actions = expert(new_obs)
        
        new_obs = torch.from_numpy(new_obs).to(config.device)
        expert_actions = torch.from_numpy(expert_actions).to(config.device)
        new_data = TensorDataset(new_obs, expert_actions)
        
        # add new data to dataset
        dataset = ConcatDataset([dataset, new_data])
            

        avrg_mean, avrg_std = Eval(config, agent_wapper(config, agent))
        print('[DAgger iter {}] r_mean: {:.2f}  r_std: {:.2f}'.format(k + 1, avrg_mean, avrg_std))

        
    return agent_wapper(config, agent)
    
def run_agent(config, agent, num_rollouts):
    env = config.env
    max_steps = env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for _ in range(num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while not done:
            action = agent(obs[None, :])
            action = action.reshape(-1)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if steps >= max_steps:
                break
        returns.append(totalr)

    avrg_mean, avrg_std = np.mean(returns), np.std(returns)
    observations = np.array(observations).astype(np.float32)
    actions = np.array(actions).astype(np.float32)

    return observations, actions, avrg_mean, avrg_std

def Eval(config, agent):
    *_, avrg_mean, avrg_std = run_agent(config, agent, config.n_eval_rollouts)

    return avrg_mean, avrg_std


def save_model(config, model, PATH):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    PATH = PATH + config.envname + '-' + 'parameters.tar'
    torch.save(model.state_dict(), PATH)
    print('model saved.')

def load_model(config, model, PATH):
    PATH = PATH + config.envname + '-' + 'parameters.tar'
    model.load_state_dict(torch.load(PATH))
    print('model loaded.')
