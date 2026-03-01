
import json
import random

def generate_production_dataset(output_path="data/processed.jsonl"):
    examples = []
    
    # ── 1. Comprehensive React Knowledge ──
    react_topics = {
        "React": "React is a JavaScript library for building user interfaces based on components.",
        "Components": "Components are independent and reusable bits of code. They serve the same purpose as JavaScript functions, but work in isolation and return HTML via JSX.",
        "JSX": "JSX stands for JavaScript XML. It allows us to write HTML in React by converting HTML-like code into React elements.",
        "Props": "Props are arguments passed into React components. They are passed via HTML attributes and are read-only.",
        "State": "The state is a built-in React object that is used to contain data or information about the component. A component's state can change over time.",
        "useState": "useState is a Hook that allows you to have state variables in functional components. You pass the initial state to this function and it returns a variable with the current state value and another function to update this value.",
        "useEffect": "useEffect is a Hook that allows you to perform side effects in function components. It is similar to componentDidMount and componentDidUpdate combined.",
        "Virtual DOM": "The virtual DOM is a programming concept where an ideal, or 'virtual', representation of a UI is kept in memory and synced with the 'real' DOM by a library such as ReactDOM.",
        "Hooks": "Hooks are functions that let you 'hook into' React state and lifecycle features from function components. They don't work inside classes.",
        "Context API": "React Context is a way to manage state globally. It can be used together with the useState Hook to share state between deeply nested components more easily than with standard props."
    }

    templates = [
        "What is {topic}?", "Explain {topic} in detail.", "How does {topic} work?", 
        "Tell me about {topic}.", "What can you say regarding {topic}?", 
        "Define {topic} for me.", "Can you help me understand {topic}?"
    ]

    for topic, answer in react_topics.items():
        for temp in templates:
            user = temp.format(topic=topic)
            examples.append({"text": f"<|im_start|>system\nYou are a helpful educational tutor.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"})

    # ── 2. Massive Math Data (Production Skill) ──
    for _ in range(3000):
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        op = random.choice(["+", "-", "*"])
        if op == "+": res = a + b
        elif op == "-": res = a - b
        else:
            a, b = random.randint(1, 100), random.randint(1, 10)
            res = a * b
        
        user = f"Calculate {a} {op} {b}"
        assistant = f"The result of {a} {op} {b} is {res}."
        examples.append({"text": f"<|im_start|>system\nYou are a helpful educational tutor.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"})

    random.shuffle(examples)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"✅ Generated {len(examples)} production-scale examples.")

if __name__ == "__main__":
    generate_production_dataset()
