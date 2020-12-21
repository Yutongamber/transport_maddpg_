import torch
from SevenPlus.Agent.BaseAgent import BaseAgent
from MLP import MLP as MLPNetwork

class Agent(BaseAgent):
    def __init__(self, config, model=None, mode='Train', rollout=None, observer=None, side=0):
        """
        Args:
            config: 配置
                type: str
            model: 网络模型
                data: <class 'Examples.CartPole.Inference.MLP.MLP'>
                type: object
            mode: Agent模式：训练/测试
                data: Train/Test
                type: str
            rollout: Rollout缓冲区对象
                data: <class 'SevenPlus.Agent.Rollout.DefaultRollout'>
                type: object
            observer: Wrapper/Env对象
                data: None
                type: object
            side: 指明哪方Agent，例: side=0 红方；side=1 蓝方
                data: 0/1/......
                type: int
        """
        super(Agent, self).__init__(model, mode, rollout, observer, side=side)

    def select_action(self, ob_id, raw_state, reward, info, done, step_done):
        """智能体选择动作
        1.调用BaseAgent基类的self.brain函数，把原始状态传入，返回采样类和value值
        2.调用轨迹处理函数traj_deal，返回action

        Args:
            ob_id: observer_id
                type: int,
            raw_state: Cartpole返回的状态
                data: [[...]]
                type: list
                shape: (1,4)
            reward: Cartpole返回的奖励
                type: float
            info: Cartpole返回的其余信息
                data: {}
                type: dict
            done: 游戏是否结束
                data: True/False
                type: bool
            step_done: 是否收集完一条episode数据
                data: True/False
                type: bool
        Returns:
            action：智能体采样的动作
                type: int
        """
        with torch.no_grad():
            logits, value = self.brain(raw_state)
            # sampling_class: Categorical(logits: torch.Size([1, 2]))
            # value: tensor([[.]])
        action = self.traj_deal(ob_id, raw_state, reward, info,
                                done, step_done, logits, value)
        return action

    def traj_deal(self, ob_id, raw_state, reward, info, done, step_done, logits, value):
        """处理数据，生成trajectory
        1.利用传入的sampling_class对动作采样
        2.利用log_prob函数计算action_log_prob
        3.将智能体的一条轨迹数据进行存储，保存相应的信息
        Args:
            ob_id: observer_id
                type: int,
            raw_state: Cartpole返回的状态
                data: [[...]]
                type: list
                shape: (1,4)
            reward: Cartpole返回的奖励
                type: float
            info: Cartpole返回的其余信息
                data: {}
                type: dict
            done: 游戏是否结束
                data: True/False
                type: bool
            step_done: 是否收集完一条episode数据
                data: True/False
                type: bool
            action_distribution_class: 采样类
                data: <class 'torch.distributions.categorical.Categorical'>
                type: object
            value: value值
                data: [[.]]
                type: torch.Tensor
                shape: torch.Size([1, 1])
        Returns:
            action：智能体采样的动作
                type: int
        """

        ##############################################################
        # # 1.利用传入的sampling_class对动作采样
        # action = action_distribution_class.sample(logits)
        # # 2.利用log_prob函数计算action_log_prob
        # action_log_prob = action_distribution_class.log_prob(action)
        ##############################################################

        action = self.gumbel_softmax(logits, hard=True)
        action_log_prob = action_distribution_class.log_prob(action) # TODO???
        # 3.将智能体的一条轨迹数据进行存储，保存相应的信息
        self.rollout.append_a_traj(ob_id, raw_state, reward, info, done,
                                   step_done, action, action_log_prob, value)
        action = action.item()
        if self.mode == "Train":
            self.rollout.append_a_traj(
                ob_id, raw_state, reward, info, done,
                step_done, action, action_log_prob, value
            )
        if self.mode != "multi_agents":
            return action
        else:
            return {
                "action": action,
                # "state" : raw_state,
                # "reward" : reward,
                # "info" : info,
                # "done" : done,
                # "step_done" : step_done,
                "action_log_prob": action_log_prob,
                "value": value
            }

        # support functions

    def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
        """Sample from Gumbel(0, 1)"""
        U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
        return -torch.log(-torch.log(U + eps) + eps)

    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    def gumbel_softmax_sample(logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits.shape, tens_type=type(logits.data))
        return F.softmax(y / temperature, dim=1)

    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
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
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = onehot_from_logits(y)
            y = (y_hard - y).detach() + y
        return y