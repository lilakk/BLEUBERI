import json

file_path = "data/mt_bench/model_judgment/gpt-4.1-mini_single.jsonl"
line_number = 0
broken_line_numbers = []
with open(file_path, 'r') as f:
    for line in f:
        line_number += 1
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error on line {line_number}: {e}")
            print(f"Content: {line.strip()}")
            # You might want to break or continue depending on if you want all errors
            import pdb; pdb.set_trace()
            broken_line_numbers.append(line_number)
            continue

print(f"Broken line numbers: {broken_line_numbers}")
print(f"Total lines: {line_number}")