import torch, torch.nn as nn

class PPOLite(nn.Module):
    """PPO-Lite: 五源异构特征 → 融合 → actor + critic"""
    def __init__(self, cfg: dict):
        super().__init__()
        c = cfg['train']          # 超参统一放 train 段
        self.hidden = c['model_hidden']        # 256 足够
        self.act_dim = cfg['env']['action_dim']  # ←←← 改这里

        # 5 路输入各自过一个小 MLP 降维
        dims = cfg['env']['state_dims']        # 由 env 上报 [scalar, bench, board, hand, shop]
        self.enc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            ) for d in dims
        ])

        # 融合后特征
        fused = 5 * 32
        self.backbone = nn.Sequential(
            nn.Linear(fused, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU()
        )

        # 头
        self.actor = nn.Linear(self.hidden, self.act_dim)          # logits
        self.critic = nn.Linear(self.hidden, 1)

    def forward(self, states):
        """
        states: list[tensor] 长度 5，每个 tensor 形状 (B, d_i)
        return logits (B, A), value (B, 1)
        """
        assert len(states) == 5
        feats = [enc(s) for enc, s in zip(self.enc, states)]
        x = torch.cat(feats, dim=1)
        x = self.backbone(x)
        return self.actor(x), self.critic(x)

    @torch.no_grad()
    def step(self, states):
        """供 trainer 调用：返回 (action, log_prob, value)"""
        logits, val = self(states)
        prob = torch.softmax(logits, -1)
        dist = torch.distributions.Categorical(prob)
        a = dist.sample()
        return a, dist.log_prob(a), val.squeeze(-1)

    def act(self, states):
        """仅用于测试：确定性 greedy"""
        logits, _ = self(states)
        return logits.argmax(-1).item()