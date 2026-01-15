import os, math, torch, torch.nn as nn, numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from src.model import PPOLite
import random


class PPOTrainer:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.t_cfg = cfg['train']
        self.device = torch.device(cfg['device'])
        self.model = PPOLite(cfg).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(self.t_cfg['lr']))
        self.scaler = GradScaler() if cfg.get('mixed_precision',False) and self.device.type=='cuda' else None
        self.best_return = -math.inf
        self._ep = 0
        self.keys = ['states','actions','logp','returns','adv']
        self.reset_memory()

    # ---------- 工具 ----------
    def reset_memory(self):
        self.memory = {k: [] for k in self.keys}

    def save(self, path, best=False):
        torch.save({'model':self.model.state_dict(),'opt':self.opt.state_dict(),
                    'episode':self._ep,'best_return':self.best_return}, path)
        if best:
            print(f'[Save] best model -> {path}')

    def load_model(self, path=None):
        path = path or os.path.join(self.t_cfg['save_dir'],'best.pth')
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.opt.load_state_dict(ckpt['opt'])
        self._ep = ckpt.get('episode',0)
        self.best_return = ckpt.get('best_return',-math.inf)

    # ---------- 采样 ----------
    @torch.no_grad()
    def collect(self, steps: int):
        state = self.env.reset()
        states = [[] for _ in state]
        rewards, dones, values, actions, logp = [], [], [], [], []

        external = [100, 1, 1, 0, 0, 0]  # 初始 dummy
        for _ in range(steps):
            # 1. 给你看当前状态
            print(f"\n[Step {_}] 当前状态 → "
                  f"HP:{external[0]:.0f}  上把胜利:{external[1]:.0f}  "
                  f"金币:{external[2]:.0f}  利息/连胜:{external[3]:.0f}  "
                  f"羁绊数:{external[4]:.0f}  总星级:{external[5]:.0f}")

            # 2. 你把新的 6 维状态敲进来（直接回车=沿用上一帧）
            external = self._read_external(external)

            # 3. 模型选动作
            st = [s.to(self.device) for s in state]
            with autocast(enabled=self.scaler is not None):
                act, lp, val = self.model.step(st)
            a_int = act.item()
            print("【模型动作】>>> " + self.env.decode_action(a_int))


            # 5. 等你回车确认后再继续（方便你抄动作去真实环境）
            input("  按回车进入下一帧... ")

            # 6. 环境步进（这里用你刚输入的 external 计算奖励）
            next_s, rew, done, _ = self.env.step(a_int, external)

            # 7. 存经验
            for i, s in enumerate(state):
                states[i].append(s.squeeze(0))
            actions.append(act)
            logp.append(lp)
            values.append(val)
            rewards.append(rew)
            dones.append(done)

            state = next_s
            if done:
                state = self.env.reset()
                external = [100, 1, 1, 0, 0, 0]


        # 8. 最后一帧的 V(s') 用于 GAE
        with autocast(enabled=self.scaler is not None):
            _, _, last_v = self.model.step([s.to(self.device) for s in state])
        return states, actions, logp, rewards, dones, values, last_v

    def _mock_external(self, ext):
        """
        训练时从 cmd 实时读取 6 个数字：
        格式：hp ifwin gold extra state_num star
        示例：92 1 30 3 2 15
        """
        while True:
            try:
                raw = input('ENV>>> ').strip()
                if raw == '':  # 直接回车就沿用上一帧
                    return ext
                parts = list(map(float, raw.split()))
                if len(parts) != 6:
                    print('需要 6 个数字：hp ifwin gold extra state_num star')
                    continue
                return parts
            except ValueError:
                print('输入非法，重新输入')

    def _read_external(self, ext):
        while True:
            try:
                raw = input('ENV>>> ').strip()
                if raw == '':
                    return ext
                parts = list(map(float, raw.split()))
                if len(parts) != 6:
                    print('需要 6 个数字：hp ifwin gold extra state_num star')
                    continue
                return parts
            except ValueError:
                print('输入非法，重新输入')

    # ---------- GAE ----------
    def compute_gae(self, rewards, values, dones, last_v, gamma=0.99, lam=0.95):
        values = values + [last_v]
        gae = 0
        adv = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step+1] * (1-dones[step]) - values[step]
            gae = delta + gamma * lam * (1-dones[step]) * gae
            adv.insert(0, gae)
        returns = [adv[i] + values[i] for i in range(len(adv))]
        return returns, adv

    # ---------- 更新 ----------
    def update(self):
        states, actions, logp_old, rewards, dones, values, last_v = self.collect(self.t_cfg['collect_steps'])
        returns, adv = self.compute_gae(rewards, values, dones, last_v,
                                        gamma=self.t_cfg['gamma'],
                                        lam=self.t_cfg.get('lambda',0.95))
        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states = [torch.stack(s).to(self.device) for s in states]
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        logp_old = torch.tensor(logp_old, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(*states, actions, logp_old, returns, adv)
        loader = DataLoader(dataset, batch_size=self.t_cfg['batch_size'],
                            shuffle=True, drop_last=True)

        for epoch in range(self.t_cfg['update_epochs']):
            for batch in loader:
                *batch_states, b_act, b_logp_old, b_ret, b_adv = batch
                with autocast(enabled=self.scaler is not None):
                    logits, val = self.model(batch_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(b_act)
                    ratio = torch.exp(logp - b_logp_old)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1-self.t_cfg['clip_eps'],
                                        1+self.t_cfg['clip_eps']) * b_adv
                    actor_loss  = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(val.squeeze(), b_ret)
                    loss = actor_loss + 0.5 * critic_loss

                self.opt.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.opt.step()

    # ---------- 训练主循环 ----------
    def train(self):
        os.makedirs(self.t_cfg['save_dir'], exist_ok=True)
        for self._ep in range(self._ep+1, self.t_cfg['total_episodes']+1):
            self.update()
            if self._ep % self.t_cfg.get('test_interval',50) == 0:
                ret = self.evaluate()
                if ret > self.best_return:
                    self.best_return = ret
                    self.save(os.path.join(self.t_cfg['save_dir'],'best.pth'), best=True)
                self.save(os.path.join(self.t_cfg['save_dir'],'latest.pth'))
                print(f'[Episode {self._ep}] test_return={ret:.2f}  best={self.best_return:.2f}')

    # ---------- 测试 ----------
    @torch.no_grad()
    def evaluate(self, episodes=3):
        ret = 0
        for _ in range(episodes):
            state = self.env.reset()
            external = [100,1,1,0,0,0]
            done = False
            while not done:
                external = self._mock_external(external)
                state = [s.to(self.device) for s in state]
                with autocast(enabled=self.scaler is not None):
                    a, _, _ = self.model.step(state)
                state, r, done, _ = self.env.step(a.item(), external)
                ret += r
        return ret / episodes
