import logging
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from config import config
from prompts import FEEDBACK_PROMPT, REVISION_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RevisionInput(BaseModel):
    system_prompt: str
    user_input: str
    initial_output: Optional[str] = None


class EvaluationResult(BaseModel):
    overall_score: float
    aspect_scores: Dict[str, float]
    combined_reasoning: str


class IterationResult(BaseModel):
    should_stop: bool = False
    stop_reason: str = ""
    log_entry: Dict[str, Any] = {}
    suggestions: List[str] = []
    revised_output: str = ""
    evaluation: Optional[EvaluationResult] = None


class RevisionResult(BaseModel):
    final_output: str
    revision_history: List[str]
    evaluation_history: List[EvaluationResult]
    final_suggestions: List[str]
    history_log: List[Dict[str, Any]]


class Reviser:
    def __init__(
            self,
            agent_llm: BaseChatModel,
            reviser_llm: BaseChatModel,
            evaluator: Any = None,
            max_iterations: int = config['reviser']['max_iterations']
    ):
        self.agent_llm = agent_llm
        self.reviser_llm = reviser_llm
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.feedback_chain = FEEDBACK_PROMPT | self.reviser_llm | StrOutputParser()
        self.revision_chain = REVISION_PROMPT | self.agent_llm | StrOutputParser()

    async def revise(self, revision_input: RevisionInput) -> RevisionResult:
        current_output = revision_input.initial_output or "No initial output provided."
        revision_history = [current_output]
        evaluation_history = []
        history_log = []

        for i in range(self.max_iterations):
            logger.info(f"Starting revision iteration {i + 1}")

            iteration_result = await self._perform_iteration(
                revision_input=revision_input,
                current_output=current_output,
                previous_output=revision_history[-2] if len(revision_history) > 1 else None,
                iteration=i + 1
            )

            history_log.append(iteration_result.log_entry)
            if iteration_result.evaluation:
                evaluation_history.append(iteration_result.evaluation)
                logger.info(f"Iteration {i + 1} - Evaluation Results:")
                logger.info(f"Overall Score: {iteration_result.evaluation.overall_score}")
                logger.info(f"Aspect Scores: {iteration_result.evaluation.aspect_scores}")
                logger.info(f"Reasoning: {iteration_result.evaluation.combined_reasoning}")

            if iteration_result.should_stop:
                logger.info(iteration_result.stop_reason)
                break

            current_output = iteration_result.revised_output
            revision_history.append(current_output)
            logger.info(f"Completed revision iteration {i + 1}")
            logger.info(f"Revised Output: {current_output}...")

        return RevisionResult(
            final_output=current_output,
            revision_history=revision_history,
            evaluation_history=evaluation_history,
            final_suggestions=iteration_result.suggestions,
            history_log=history_log
        )

    async def _perform_iteration(self,
                                 revision_input: RevisionInput,
                                 current_output: str,
                                 previous_output: Optional[str],
                                 iteration: int) -> IterationResult:

        evaluation = await self._evaluate(
            system_prompt=revision_input.system_prompt,
            user_input=revision_input.user_input,
            current_output=current_output,
            previous_output=previous_output
        ) if self.evaluator else None

        if evaluation and evaluation.overall_score >= config['reviser']['target_score']:
            return IterationResult(
                should_stop=True,
                stop_reason="Target score reached",
                evaluation=evaluation
            )

        feedback = await self._get_feedback(
            system_prompt=revision_input.system_prompt,
            user_input=revision_input.user_input,
            current_output=current_output,
            previous_output=previous_output,
            evaluation=evaluation
        )
        revision_result = await self._get_revision(
            system_prompt=revision_input.system_prompt,
            user_input=revision_input.user_input,
            current_output=current_output,
            previous_output=previous_output,
            evaluation=evaluation,
            feedback=feedback
        )

        suggestions, revised_output = await self.parse_revision_result(revision_result)

        log_entry = {
            "iteration": iteration,
            "evaluation": evaluation.dict() if evaluation else None,
            "feedback": feedback,
            "suggestions": suggestions,
            "revised_output": revised_output
        }

        if revised_output == current_output:
            return IterationResult(
                should_stop=True,
                stop_reason="Output converged",
                log_entry=log_entry,
                suggestions=suggestions,
                revised_output=revised_output,
                evaluation=evaluation
            )

        return IterationResult(
            log_entry=log_entry,
            suggestions=suggestions,
            revised_output=revised_output,
            evaluation=evaluation
        )

    async def _evaluate(self,
                        system_prompt: str,
                        user_input: str,
                        current_output: str,
                        previous_output: Optional[str]) -> Optional[EvaluationResult]:
        if not self.evaluator:
            return None
        result = await self.evaluator.evaluate(system_prompt, user_input, current_output, previous_output)
        return EvaluationResult(
            overall_score=result.overall_score,
            aspect_scores=result.aspect_scores,
            combined_reasoning=result.combined_reasoning
        )

    async def _get_feedback(self,
                            system_prompt: str,
                            user_input: str,
                            current_output: str,
                            previous_output: Optional[str],
                            evaluation: Optional[EvaluationResult]) -> str:
        feedback_input = {
            "system_prompt": system_prompt,
            "user_input": user_input,
            "current_output": current_output,
            "previous_output": previous_output or "No previous output available.",
        }
        if evaluation:
            feedback_input.update({
                "evaluation_overall_score": evaluation.overall_score,
                "evaluation_aspect_scores": evaluation.aspect_scores,
                "evaluation_combined_reasoning": evaluation.combined_reasoning
            })
        return await self.feedback_chain.ainvoke(feedback_input)

    async def _get_revision(self,
                            system_prompt: str,
                            user_input: str,
                            current_output: str,
                            previous_output: Optional[str],
                            evaluation: Optional[EvaluationResult],
                            feedback: str) -> str:
        revision_input = {
            "system_prompt": system_prompt,
            "user_input": user_input,
            "current_output": current_output,
            "previous_output": previous_output or "No previous output available.",
            "feedback": feedback
        }
        if evaluation:
            revision_input.update({
                "evaluation_overall_score": evaluation.overall_score,
                "evaluation_aspect_scores": evaluation.aspect_scores,
                "evaluation_combined_reasoning": evaluation.combined_reasoning
            })
        return await self.revision_chain.ainvoke(revision_input)

    @staticmethod
    async def parse_revision_result(revision_result: str) -> Tuple[List[str], str]:
        suggestions = []
        revised_output = ""
        current_section = None

        for line in revision_result.split('\n'):
            line = line.strip()
            if line.startswith('REVISED OUTPUT:'):
                current_section = 'revised_output'
                continue
            elif line.startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
                continue

            if current_section == 'suggestions':
                suggestions.append(line)
            elif current_section == 'revised_output':
                revised_output += line + '\n'

        return suggestions, revised_output.strip()
