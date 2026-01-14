from src.utils import load_config, encode_yaml
from src.env import TFTEnv
from src.trainer import PPOTrainer
import argparse, torch

def build_token_maps(cfg: dict):
    data = cfg['data']
    hero_cost  = {k: v['cost']  for k, v in data['heroes'].items()}
    hero_trait = {k: v['traits'] for k, v in data['heroes'].items()}
    trait_levels = {k: v['levels'] for k, v in data['traits'].items()}
    shop_names = cfg['env']['selectable_heros']
    return {
        'hero_cost': hero_cost,
        'hero_trait': hero_trait,
        'trait_levels': trait_levels,
        'shop_names': shop_names,
    }

def main():
    parser = argparse.ArgumentParser(description="TFT-PPO 最小先验版")
    parser.add_argument("--mode", default="train", choices=["train","test","encode"])
    parser.add_argument("--config", default="config/mini.yaml")
    args = parser.parse_args()

    if args.mode == "encode":
        encode_yaml(args.config, args.config.replace('.yaml', '_encoded.yaml'))
        return

    cfg = load_config(args.config.replace('.yaml', '_encoded.yaml'))
    cfg['static_config_path'] = args.config.replace('.yaml', '_encoded.yaml')
    cfg['token_maps'] = build_token_maps(cfg)

    env = TFTEnv(cfg)
    trainer = PPOTrainer(env, cfg)

    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        trainer.load_model()
        trainer.test()

if __name__ == "__main__":
    main()