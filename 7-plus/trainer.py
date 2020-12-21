import torch
import numpy as np
from SevenPlus.Train.BaseTrainer import Trainer
from ddpg import DDPGAgent

class MADDPGTrainer(Trainer):
    def __init__(self, agents, buffer, net, decay_rate, action_space, state_space):
        super().__init__(agents, buffer)

        self.net = net
        self.gamma = 0.98
        self.tau = 0.01
        self.lr = 0.01
        self.discrete_action = True
        self.hidden_dim = 64

        self.agent_init_params = [{'num_in_pol': 128, 'num_out_pol': 5, 'num_in_critic': 266},
        {'num_in_pol': 128, 'num_out_pol': 5, 'num_in_critic': 266}]

        self.nagents = 2  # TODO
        self.alg_types = 'MADDPG'
        # self.agents = [DDPGAgent(lr=self.lr, discrete_action=self.discrete_action,
        #                          hidden_dim=self.hidden_dim,
        #                          **params)
        #                for params in agent_init_params]  # TODO

        self.agent_init_params = agent_init_params

        # 设备设定
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    def get_data(self):
        '''
        注意：当只有一个智能体，即当head_num=1时，返回数据维度为(M,T,...)
              当多余一个智能体，即当head_num>1时，返回数据维度为(N,M,T,...)
              其中N表示智能体数量，M表示环境数量，T为step数量

        即当head_num>1时的Returns
        Returns:
            states: 状态
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T,...)
            actions: 动作
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T)
            rewards: 奖励
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T)
            terminals: 终止
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T)
            values: 值函数值
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T)
            log_probs: log概率
                data: [[...],[...],...]
                type: numpy.ndarray
                shape: (M,T)

        即当head_num>1时的Returns
        Returns:
            states: 状态
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T,...)
            actions: 动作
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T)
            rewards: 奖励
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T)
            terminals: 终止
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T)
            values: 值函数值
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T)
            log_probs: log概率
                data: [[[...],[...],...],...]
                type: numpy.ndarray
                shape: (N,M,T)

        '''

        head_num = TRAJ_HEAD_NUM if not ENABLE_MULTI_AGENT else MULTI_AGENTS_NUM
        print("**************", head_num)
        states = [getattr(self.buffer, "actor_{}_states".format(i)).get_data() for i in range(
            head_num)] if head_num > 1 else self.buffer.actor_0_states.get_data()
        actions = [getattr(self.buffer, "actor_{}_actions".format(i)).get_data() for i in range(
            head_num)] if head_num > 1 else self.buffer.actor_0_actions.get_data()
        rewards = [getattr(self.buffer, "actor_{}_rewards".format(i)).get_data(
        ) for i in range(head_num)] if head_num > 1 else self.buffer.actor_0_rewards.get_data()
        terminals = self.buffer.dones.get_data()
        values = [getattr(self.buffer, "actor_{}_values".format(i)).get_data(
        ) for i in range(head_num)] if head_num > 1 else self.buffer.actor_0_values.get_data()
        log_probs = [getattr(self.buffer, "actor_{}_logprobs".format(i)).get_data() for i in range(
            head_num)] if head_num > 1 else self.buffer.actor_0_logprobs.get_data()
        print("********buffer数据**********")
        print("state_list: ", "shape:", np.array(states).shape)  # 3 x len(并行环境) x len(traj)
        print("action_list: ", "shape:", np.array(actions).shape)
        print("reward_list: ", "shape:", np.array(rewards).shape)
        print("terminal_list:", "shape:", np.array(terminals).shape)
        print("value_list: ", "shape:", np.array(values).shape)
        print("log_prob_list: ", "shape:", np.array(log_probs).shape)
        return states, actions, rewards, terminals, values, log_probs

    def step(self):
        self._checkBuffer()
        # self.data_length = len(self.buffer.rewards_0.get_data()[1])
        tmp = self.buffer.dones.get_storage()
        if len(tmp) > 0:
            self.data_length = len(tmp[0])
        if self.data_length >= 1:
            # print("数据消费：", self.data_length)
            model = self._update()
            self.data_length = 0
            # self.set_policy(model)
            return model
        else:
            # time.sleep(3)
            return None

    def _update(self):
        obs, acs, rews, next_obs, dones = self.get_data() #TODO
        #TODO 可并行
        #TODO sample

        for a_i in range(len(self.net)):
            self.update(a_i, data, logger=logger)
        return self.net

    def update(self, agent_i, data, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """

        curr_agent = self.net[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG':
            if self.discrete_action: # one-hot encode action
                all_trgt_acs = [self.onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)]
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs, *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)

        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()
        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = self.gumbel_softmax(curr_pol_out, hard=True) # softmax + argmax

        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def calculate_return(self):
        """return计算"""
        pass

    def data_to_tensor(self):
        """数据类型转换array->tensor， # N:环境数，T：step数"""
        pass

    def loss_fn(self):
        """定义损失函数"""
        pass

    def _apply_grad(self):
        pass

    def _compute_grad(self):
        pass

    def set_dacay_rate(self, rate):
        self.decay_rate = rate

    def onehot_from_logits(logits, eps=0.0):
        """
        Given batch of logits, return one-hot sample using epsilon greedy strategy
        (based on given epsilon)
        """
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs
        # get random actions in one-hot form
        rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
            range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
        # chooses between best and random actions using epsilon greedy
        return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                            enumerate(torch.rand(logits.shape[0]))])

    def gumbel_softmax(logits, temperature=1.0, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = onehot_from_logits(y)
            y = (y_hard - y).detach() + y
        return y