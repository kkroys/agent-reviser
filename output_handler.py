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
            f.write(f"ITERATION {entry['iteration']}\n")
            f.write(f"{'='*50}\n\n")

            f.write("EVALUATION:\n")
            f.write(f"Overall Score: {entry['evaluation']['overall_score']}\n")
            f.write(f"Aspect Scores: {entry['evaluation']['aspect_scores']}\n")
            f.write(f"Reasoning:\n{entry['evaluation']['reasoning']}\n\n")

            f.write(f"FEEDBACK:\n{entry['feedback']}\n\n")

            f.write("SUGGESTIONS:\n")
            for suggestion in entry['suggestions']:
                f.write(f"- {suggestion}\n")
            f.write("\n")

            f.write(f"REVISED OUTPUT:\n{entry['revised_output']}\n\n")

            f.write(f"{'='*50}\n\n")
