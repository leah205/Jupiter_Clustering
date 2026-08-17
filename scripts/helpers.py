from config.config import cf

def get_dir_path(config):
    return f"{cf["input"]}/{config.source[:10]}/{config.source}/{cf["sys"]}"
