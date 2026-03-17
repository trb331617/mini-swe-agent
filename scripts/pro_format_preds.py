import json

input_file = 'pro_122b_preds.json'
output_file = 'formatted_patches.json'

with open(input_file, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

formatted_list = []

for instance_id, info in raw_data.items():
    patch_content = info.get('model_patch', '')

    original_prefix = info.get('model_name_or_path', 'mini-swe-agent')
    safe_prefix = original_prefix.replace('/', '_').replace(':', '_')

    formatted_list.append({
        "instance_id": instance_id,
        "patch": patch_content,
        "prefix": safe_prefix
    })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(formatted_list, f, indent=2, ensure_ascii=False)

print(f"转换成功!")
