import json
import os

def preprocess_data(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return 0

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_data = []
    current_block = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if any(skip in line for skip in ["Jump to content", "Main menu", "Toggle", "move to sidebar", "hide"]):
            continue
            
        current_block.append(line)
        
        if len(current_block) >= 10:
            context = " ".join(current_block)
            text = f"<|im_start|>system\nYou are a helpful mathematics tutor.<|im_end|>\n<|im_start|>user\nSummarize and explain these mathematical details: {context}<|im_end|>\n<|im_start|>assistant\nHere is an educational summary of those concepts: {context}<|im_end|>"
            processed_data.append({"text": text})
            current_block = []

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Created {len(processed_data)} training examples in {output_file}")
    return len(processed_data)

if __name__ == "__main__":
    preprocess_data("data/raw_math_wiki.txt", "data/processed.jsonl")
