import json
import time
import re
from groq import Groq

client = Groq(api_key='')

def parse_qa_response(raw):
    # clean markdown code blocks
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    
    # try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # try to extract JSON array with regex
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # if all fails, return empty
    print(f"Could not parse JSON: {raw[:200]}")
    return []


def generate_qa(text, title, n=3):
    prompt = f"""
You are an expert in marine biology. Based on the following article, generate {n} question-answer pairs.

Article title: {title}
Article content: {text[:3000]}

Return ONLY a JSON array like this, no extra text, no explanation:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", #"llama-3.3-70b-versatile"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    raw = response.choices[0].message.content
    return parse_qa_response(raw) 

def generate_qa_dataset(ds_split):
    dataset = []
    for i, article in enumerate(ds_split):
        try:
            pairs = generate_qa(article["clean_content"], article["title"])
            for pair in pairs:
                dataset.append({
                    "instruction": pair["question"],
                    "response": pair["answer"],
                    "source": article["title"]
                })
            time.sleep(1)
            if i % 10 == 0:
                print(f"Progress: {i}/{len(ds_split)}")
        except Exception as e:
            print(f"Skipped {article['title']}: {e}")
    return dataset
