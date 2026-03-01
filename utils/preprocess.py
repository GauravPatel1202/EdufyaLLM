import json
import os
import re

def preprocess_data(input_file, output_file):
    """
    Improved preprocessor that creates high-quality Q&A training pairs
    from scraped website content.
    """
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return 0

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Clean text
    lines = raw_text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip navigation/boilerplate
        skip_patterns = [
            "Jump to content", "Main menu", "Toggle", "move to sidebar",
            "hide", "Navigation menu", "Sign in", "Create account",
            "Read more", "External links", "References", "See also",
            "Cookie", "Privacy policy", "Terms of Use", "Wikipedia",
            "Retrieved from", "[edit]", "This article", "You can help",
        ]
        if any(skip in line for skip in skip_patterns):
            continue
        if len(line) < 10:  # Skip very short lines (likely headers/noise)
            continue
        cleaned_lines.append(line)

    processed_data = []

    # Strategy 1: Create summary/explanation pairs from paragraphs
    paragraphs = []
    current_para = []
    for line in cleaned_lines:
        current_para.append(line)
        if len(current_para) >= 5:
            paragraphs.append(" ".join(current_para))
            current_para = []
    if current_para:
        paragraphs.append(" ".join(current_para))

    question_templates = [
        ("Explain the following concept:", "Here is a clear explanation:"),
        ("What does this mean?", "Let me explain this in detail:"),
        ("Summarize the following:", "Here is a concise summary:"),
        ("Can you teach me about this?", "Of course! Here's what you need to know:"),
        ("Help me understand this:", "Sure, let me break it down:"),
    ]

    for i, para in enumerate(paragraphs):
        if len(para) < 30:  # Skip very short paragraphs
            continue

        template_idx = i % len(question_templates)
        q_prefix, a_prefix = question_templates[template_idx]

        # Q&A where context is the question
        text = (
            f"<|im_start|>system\nYou are a helpful educational tutor.<|im_end|>\n"
            f"<|im_start|>user\n{q_prefix} {para[:200]}<|im_end|>\n"
            f"<|im_start|>assistant\n{a_prefix} {para}<|im_end|>"
        )
        processed_data.append({"text": text})

    # Strategy 2: Create keyword-based Q&A from sentences
    all_text = " ".join(cleaned_lines)
    sentences = [s.strip() for s in re.split(r'[.!?]+', all_text) if len(s.strip()) > 30]

    for i in range(0, len(sentences) - 1, 2):
        sent = sentences[i]
        # Extract key phrase from beginning
        words = sent.split()
        if len(words) >= 1:
            topic = " ".join(words[:5])
            text = (
                f"<|im_start|>system\nYou are a helpful educational tutor.<|im_end|>\n"
                f"<|im_start|>user\nTell me about {topic}<|im_end|>\n"
                f"<|im_start|>assistant\n{sent.strip()}.<|im_end|>"
            )
            processed_data.append({"text": text})

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Created {len(processed_data)} high-quality training examples in {output_file}")
    return len(processed_data)

if __name__ == "__main__":
    preprocess_data("data/raw_scraped.txt", "data/processed.jsonl")
