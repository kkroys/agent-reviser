import logging
from dataclasses import dataclass
from typing import List, Optional
from langchain_core.output_parsers import StrOutputParser
from config import config
from prompts import FEEDBACK_PROMPT, REVISION_PROMPT
from tracing import tracer
from evaluator import MultiEvaluator, AggregatedEvaluationResult

logging.basicConfig(level=config['logging']['level'], format=config['logging']['format'])
logger = logging.getLogger(__name__)


@dataclass
class RevisionInput:
    system_prompt: str
    user_input: str
    initial_output: Optional[str] = None


@dataclass
class RevisionResult:
    final_output: str
    revision_history: List[str]
    evaluation_history: List[AggregatedEvaluationResult]

class Reviser:
    def __init__(self, agent_llm, reviser_llm, evaluator: MultiEvaluator, max_iterations: int = config['reviser']['max_iterations']):
        self.agent_llm = agent_llm
        self.reviser_llm = reviser_llm
        self.evaluator = evaluator
        self.max_iterations = max_iterations

        self.feedback_chain = FEEDBACK_PROMPT | self.reviser_llm | StrOutputParser()
        self.revision_chain = REVISION_PROMPT | self.agent_llm | StrOutputParser()

    @tracer(run_type="chain", name="Revision Process")
    async def revise(self, revision_input: RevisionInput) -> RevisionResult:
        current_output = revision_input.initial_output or "No initial output provided."
        revision_history = [current_output]
        evaluation_history = []
        previous_output = None

        for i in range(self.max_iterations):
            try:
                logger.info(f"Starting revision iteration {i + 1}")

                evaluation = await self.evaluator.evaluate(
                    revision_input.system_prompt,
                    revision_input.user_input,
                    current_output,
                    previous_output
                )
                evaluation_history.append(evaluation)
                logger.info(f"Overall evaluation score: {evaluation.overall_score}")
                logger.info(f"Aspect scores: {evaluation.aspect_scores}")
                logger.info(f"Combined reasoning: {evaluation.combined_reasoning}")

                if evaluation.overall_score >= config['reviser']['target_score']:
                    logger.info(f"Target score reached. Stopping revision process.")
                    break

                feedback = self.feedback_chain.invoke({
                    "system_prompt": revision_input.system_prompt,
                    "user_input": revision_input.user_input,
                    "current_output": current_output,
                    "previous_output": previous_output or "No previous output available.",
                    "evaluation_overall_score": evaluation.overall_score,
                    "evaluation_aspect_scores": evaluation.aspect_scores,
                    "evaluation_combined_reasoning": evaluation.combined_reasoning
                })

                current_output = self.revision_chain.invoke({
                    "system_prompt": revision_input.system_prompt,
                    "user_input": revision_input.user_input,
                    "current_output": current_output,
                    "previous_output": previous_output or "No previous output available.",
                    "evaluation_overall_score": evaluation.overall_score,
                    "evaluation_aspect_scores": evaluation.aspect_scores,
                    "evaluation_combined_reasoning": evaluation.combined_reasoning,
                    "feedback": feedback
                })

                previous_output = current_output
                revision_history.append(current_output)
                logger.info(f"Revision iteration {i + 1} complete")

            except Exception as e:
                logger.error(f"Error during revision iteration {i + 1}: {str(e)}")
                break

        return RevisionResult(final_output=current_output, revision_history=revision_history, evaluation_history=evaluation_history)
