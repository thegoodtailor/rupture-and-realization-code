from witnessed_ph import analyse_conversation_from_json, print_self_report
import json

with open('data/test_conversation.json', encoding='utf-8') as f:
    conv = json.load(f)

print(f"Analysing: {conv['title']}")
print(f"Turns: {len(conv['turns'])}")
print()

diagrams, graph, metrics = analyse_conversation_from_json(
    conv,
    cumulative=True,
    verbose=True
)

print()
print(print_self_report(graph, metrics))
