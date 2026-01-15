import torch, random
from src.utils import load_config

class TFTEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.board_shape = [4, 8]
        self.bench_max   = 9
        self.hand_max    = 10
        self.shop_size   = cfg['env']['shop_size']
        self.board_size = 4 * 8
        self.upgrade_idx = self.shop_size              # 升人口动作固定索引
        self.action_dim  = self.shop_size + 1 + self.board_size   # 买 + 升级 + 卖

        # 内部只维护“上一帧外部输入”，不再自己算血量/金币
        self.last_external = [100,1,1,0,0,0]
        # ===== 关键：把算好的维度写回 cfg =====
        cfg['env']['action_dim'] = self.action_dim
        cfg['env']['state_dims'] = [int(s.shape[-1]) for s in self._build_dummy_state()]

    # ---------- 标准接口 ----------
    def reset(self):
        self.last_external = [100,1,1,0,0,0]
        return self._build_state(self.last_external)

    # ---------- 内部工具 ----------
    def _build_dummy_state(self):
        """返回 5 个 (1, dim) 的 dummy tensor，仅用于量尺寸"""
        return [
            torch.randn(1, 10),                          # scalar
            torch.randn(1, self.bench_max),              # bench
            torch.randn(1, self.board_size),             # board
            torch.randn(1, self.hand_max),               # hand
            torch.randn(1, self.shop_size),              # shop
        ]

    def step(self, action: int, external: list):
        reward = self._apply_action(action, external)
        reward += self._compute_reward(*external[1:])
        done = external[0] <= 0
        self.last_external = external
        return self._build_state(external), reward, done, {}

    def get_current_state(self, verbose=False):
        txt = f"HP={self.last_external[0]} Win={self.last_external[1]} Gold={self.last_external[2]}"
        if verbose:
            print(txt)
        return txt

    # ---------- 内部 ----------
    def _apply_action(self, action, external):
        gold = external[2]
        BUY_BONUS = 1.0 # 应该为1
        if action < self.shop_size:
            token = self.cfg['token_maps']['shop_names'][action]
            cost = self.cfg['token_maps']['hero_cost'][token]
            if gold >= cost:
                external[2] -= cost
                return (cost - 1) * 0.1 * BUY_BONUS
        elif action == self.upgrade_idx:  # 升人口
            if gold >= 4:
                external[2] -= 4
                return 0.2
        else:  # 卖/移动
            return 0.05

    def _compute_reward(self, ifwin, gold, extra, state_num, star):
        # --- 1. 胜利 >> 一切 ---
        win_bonus = 5.0 if ifwin else -8.0  # 输一把狠罚

        # --- 2. 血量惩罚随线性加深 ---
        hp_penalty = (100 - self.last_external[0]) * 0.1  # 每少一点血扣 0.1

        # --- 3. 羁绊 & 星级给正反馈 ---
        trait_bonus = state_num * 1.5
        star_bonus = star * 0.3

        # --- 4. 利息/连胜即时甜头 ---
        extra_bonus = extra * 1.0

        total = win_bonus - hp_penalty + trait_bonus + star_bonus + extra_bonus
        return total

    def _build_state(self, ext):
        hp, ifwin, gold, extra, state_num, star = ext
        scalar = [hp/100, ifwin, gold/50, extra/10, state_num/10, star/30] + [0]*4
        return [torch.tensor([scalar], dtype=torch.float32),
                torch.zeros(1, self.bench_max),
                torch.zeros(1, self.board_size),
                torch.zeros(1, self.hand_max),
                torch.zeros(1, self.shop_size)]

    def decode_action(self, a: int) -> str:
        if a < self.shop_size:
            hero = self.cfg['token_maps']['shop_names'][a]
            cost = self.cfg['token_maps']['hero_cost'][hero]
            return f"购买商店第{a + 1}张卡 '{hero}'（花费{cost}金币）"
        elif a == self.upgrade_idx:
            return "升人口（花费4金币）"
        else:
            pos = a - self.upgrade_idx - 1
            return f"卖出/移动棋盘位置{pos}"