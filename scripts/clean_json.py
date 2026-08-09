import json

def remove_keys(obj, keys_to_remove):
    """
    Recursively removes all occurrences of any key in keys_to_remove.
    """
    keys_to_remove = set(keys_to_remove)

    def _remove(item):
        if isinstance(item, dict):
            return {
                key: _remove(value)
                for key, value in item.items()
                if key not in keys_to_remove
            }
        elif isinstance(item, list):
            return [_remove(element) for element in item]
        else:
            return item

    return _remove(obj)


def remove_keys_from_json(input_file, output_file, keys_to_remove):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = remove_keys(data, keys_to_remove)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4)

if __name__ == "__main__":
    remove_keys_from_json(
        "mahalanobis_clusters.json",
        "new_mahalanobis_clusters.json",
        [
            "20251016UTc-SED",
            "20251016UTf-SED",
            "NH3_PCld_AOI_CI",
            "AOI_CI"

        ]
    )