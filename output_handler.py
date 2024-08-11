import os
from typing import List, Dict, Any


def write_output_files(final_output: str,
                       history_log: List[Dict[str, Any]],
                       debug: bool = False):
    if not debug:
        return

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'final_output.txt'), 'w') as f:
        f.write(final_output)

    with open(os.path.join(output_dir, 'revision_history.txt'), 'w') as f:
        for entry in history_log:
            f.write(f"{'='*50}\n")
            f.write(f"ITERATION {entry.get('iteration', 'N/A')}\n")
            f.write(f"{'='*50}\n\n")

            if 'evaluation' in entry and entry['evaluation']:
                f.write("EVALUATION:\n")
                f.write(f"Overall Score: {entry['evaluation'].get('overall_score', 'N/A')}\n")
                f.write(f"Aspect Scores: {entry['evaluation'].get('aspect_scores', 'N/A')}\n")
                f.write(f"Reasoning:\n{entry['evaluation'].get('combined_reasoning', 'N/A')}\n\n")
            else:
                f.write("EVALUATION: Not available\n\n")

            f.write(f"FEEDBACK:\n{entry.get('feedback', 'N/A')}\n\n")

            f.write("SUGGESTIONS:\n")
            for suggestion in entry.get('suggestions', []):
                f.write(f"- {suggestion}\n")
            f.write("\n")

            f.write(f"REVISED OUTPUT:\n{entry.get('revised_output', 'N/A')}\n\n")

            f.write(f"{'='*50}\n\n")


def write_structured_output(history_log: List[Dict[str, Any]], output_file: str = 'structured_output.md'):
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, output_file), 'w') as f:
        f.write("# Revision Process Output\n\n")
        f.write("## Revision History\n\n")

        for entry in history_log:
            f.write(f"### Iteration {entry.get('iteration', 'N/A')}\n\n")

            if 'evaluation' in entry and entry['evaluation']:
                f.write("#### Evaluation\n\n")
                f.write(f"- Overall Score: {entry['evaluation'].get('overall_score', 'N/A')}\n")
                f.write("- Aspect Scores:\n")
                for aspect, score in entry['evaluation'].get('aspect_scores', {}).items():
                    f.write(f"  - {aspect}: {score}\n")
                f.write("\n")

                f.write("#### Reasoning\n\n")
                f.write(f"{entry['evaluation'].get('combined_reasoning', 'N/A')}\n\n")

            f.write("#### Suggestions\n\n")
            for suggestion in entry.get('suggestions', []):
                f.write(f"- {suggestion}\n")
            f.write("\n")

            f.write("#### Revised Output\n\n")
            f.write(f"```\n{entry.get('revised_output', 'N/A')}\n```\n\n")

            if entry != history_log[-1]:
                f.write("---\n\n")  # Separator between iterations

    print(f"Structured output has been written to {os.path.join(output_dir, output_file)}")
