import json
from rag_chain import answer_query

# Sample evaluation pairs (curated manually from the scraped docs)
eval_set = [
    {"question": "What is a Kubernetes Pod?", "expected_snippet": "smallest deployable unit"},
    {"question": "How do I expose a service externally?", "expected_snippet": "NodePort or LoadBalancer"},
    # Add more...
]

def run_eval():
    results = []
    for item in eval_set:
        ans, srcs = answer_query(item["question"])
        # Simple check if expected snippet appears (can be improved)
        found = item["expected_snippet"].lower() in ans.lower()
        results.append({
            "question": item["question"],
            "answer": ans,
            "expected_found": found
        })
    accuracy = sum(r["expected_found"] for r in results) / len(results)
    print(f"Accuracy (simple substring match): {accuracy:.2%}")
    return results

if __name__ == "__main__":
    run_eval()