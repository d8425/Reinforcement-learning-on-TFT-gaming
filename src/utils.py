import yaml, torch, os, hashlib

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def encode_yaml(src_yaml, dst_yaml):
    cfg = load_config(src_yaml)
    cfg['_encoded'] = True
    cfg['_hash']  = hashlib.md5(open(src_yaml,'rb').read()).hexdigest()[:8]
    with open(dst_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)