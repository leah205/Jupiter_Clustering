from config.config import cf
import json


def save_stats_json(config, stats):
    keywords = config.map.keywords
    keyword_str = '_'.join(keywords)

    with open(f"{cf["json"]}") as f:
            data = json.load(f)
            dim_obj = (data
                .setdefault(config.map.name, {})
                .setdefault(keyword_str, {}))
            
            dim_obj[str(config.cluster.n_comp)] = stats
    
    with open(f"{cf["json"]}", "w") as f:
        
        json.dump(data, f, indent=2)