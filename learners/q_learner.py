import copy
import torch

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.heads.mlp import MLPHead
from utils.loss import BYOLLoss


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None

        if args.mixer is not None:

            if args.mixer == "qmix":
                self.mixer = QMixer(args)

            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # For Self-Supervised Learning (Masked Reconstruction)
        if args.ssl_on:
            self.ssl_loss = BYOLLoss()

            # Define Momentum Mac
            self.momentum_mac = copy.deepcopy(mac)

            # Define Online Projector & Predictor
            self.projector = MLPHead(
                in_features=args.rnn_hidden_dim,
                out_features=args.rnn_hidden_dim
            )
            self.predictor = MLPHead(
                in_features=args.rnn_hidden_dim,
                out_features=args.rnn_hidden_dim
            )

            # Define Momentum Projector
            self.momentum_projector = MLPHead(
                in_features=args.rnn_hidden_dim,
                out_features=args.rnn_hidden_dim
            )

            self._initialize_momentum_net()

            # Add Online Projector & Predictor Parameters
            self.params += list(self.projector.parameters())
            self.params += list(self.predictor.parameters())

            self.optimiser = torch.optim.RMSprop(
                params=self.params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps)

        else:  # Only Multi-Agent Reinforcement Learning
            self.optimiser = torch.optim.RMSprop(
                params=self.params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"] # number of actions 

    
        mac_out = []
        hidden_states = []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            hidden_states.append(self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

        mac_out = torch.stack(mac_out, dim=1)  # 
        hidden_states = torch.stack(hidden_states, dim=1)

        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        target_mac_out = []
        target_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_hidden_states.append(self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

        target_mac_out = torch.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_hidden_states = torch.stack(target_hidden_states[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)

            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999

            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]

            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            Q_total = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_Q_total = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        else:
            Q_total = chosen_action_qvals
            target_Q_total = target_max_qvals

        # 1 step Q 
        targets = rewards + self.args.gamma * (1 - terminated) * target_Q_total

        # Td-error
        td_error = (Q_total - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # For Self-Supervised Learning (Masked Reconstruction)
        if self.args.ssl_on:
            total_online_hidden_states = []
            self.mac.init_hidden(batch.batch_size)

            for t in range(batch.max_seq_length):
                online_hidden_states = self.mac.online_forward(batch, t=t)
                total_online_hidden_states.append(
                    online_hidden_states.view(batch.batch_size, self.args.n_agents, -1))

            total_online_hidden_states = torch.stack(total_online_hidden_states, dim=1)

            # Momentum
            total_momentum_hidden_states = []
            self.momentum_mac.init_hidden(batch.batch_size)

            for t in range(batch.max_seq_length):
                momentum_hidden_states = self.momentum_mac.momentum_forward(batch, t=t)
                total_momentum_hidden_states.append(
                    momentum_hidden_states.view(batch.batch_size, self.args.n_agents, -1))

            total_momentum_hidden_states = torch.stack(total_momentum_hidden_states, dim=1)

            # Online projector & predictor
            projection = self.projector(total_online_hidden_states)
            prediction = self.predictor(projection)

            # Momentum projector
            with torch.no_grad():
                # self.momentum_projector.eval()
                momentum_projection = self.momentum_projector(total_momentum_hidden_states)

            ssl_loss = self.ssl_loss.calculate_loss(pred=prediction, true=momentum_projection)
            ssl_mean_loss = ssl_loss.mean()
            loss += ssl_mean_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self._update_momentum_net()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (Q_total * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)

        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.logger.console_logger.info("Updated target network")

    def _initialize_momentum_net(self):
        for param_q, param_k in zip(self.mac.parameters(), self.momentum_mac.parameters()):
            param_k.data.copy_(param_q.data)  # update
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_k.data.copy_(param_q.data)  # update
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_momentum_net(self):
        """
        Exponential Moving Average Update (Same as MoCo Momentum Update)
        """
        print("momentum update!")
        momentum = self.args.momentum
        for param_q, param_k in zip(self.mac.parameters(), self.momentum_mac.parameters()):
            param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

        for param_q, param_k in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

        if self.args.ssl_on:
            self.momentum_mac.cuda()
            self.projector.cuda()
            self.predictor.cuda()
            self.momentum_projector.cuda()

        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)

        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))

        torch.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)

        if self.mixer is not None:
            self.mixer.load_state_dict(
                torch.load("{}/mixer.th".format(path),
                           map_location=lambda storage, loc: storage))

        self.optimiser.load_state_dict(
            torch.load("{}/opt.th".format(path),
                       map_location=lambda storage, loc: storage))
