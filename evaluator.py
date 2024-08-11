import logging
import re

import aiohttp
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from statistics import mean
import asyncio

from config import config
from prompts import EVALUATION_SYSTEM_PROMPTS, EVALUATION_USER_PROMPT


class EvaluationResult(BaseModel):
    score: int = Field(...,
                       ge=config['evaluation']['score_range']['min'],
                       le=config['evaluation']['score_range']['max'])
    reasoning: str


class BaseEvaluator:
    async def evaluate(self,
                       system_prompt: str,
                       user_input: str,
                       current_output: str,
                       previous_output: Optional[str] = None) -> EvaluationResult:
        raise NotImplementedError


class OpenAIEvaluator(BaseEvaluator):
    def __init__(self,
                 api_key: str,
                 model: str,
                 evaluation_aspect: str,
                 api_base: str = "https://api.openai.com/v1/chat/completions"):
        self.api_key = api_key
        self.model = model
        self.evaluation_aspect = evaluation_aspect
        self.api_base = api_base
        self.logger = logging.getLogger(__name__)

    async def evaluate(self,
                       system_prompt: str,
                       user_input: str,
                       current_output: str,
                       previous_output: Optional[str] = None) -> EvaluationResult:
        evaluation_context = {
            "aspect": self.evaluation_aspect,
            "system_prompt": system_prompt,
            "user_input": user_input,
            "current_output": current_output,
            "previous_output": previous_output or "No previous output available."
        }

        user_prompt = EVALUATION_USER_PROMPT.format(**evaluation_context)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.api_base,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=self._prepare_api_request(user_prompt)
                ) as response:
                    result = await response.json()

            response_text = result['choices'][0]['message']['content']
            return await self.parse_evaluation_response(response_text)
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise

    def _prepare_api_request(self, user_prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPTS[self.evaluation_aspect]},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": config['openai']['temperature'],
            "max_tokens": config['openai']['max_tokens']
        }

    @staticmethod
    async def parse_evaluation_response(response_text: str) -> EvaluationResult:
        score_match = re.search(r'(?i)score:?\s*(\d+)', response_text)
        score = int(score_match.group(1)) if score_match else 0

        reasoning_match = re.search(r'(?i)reasoning:?\s*(.*)', response_text, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # If no "Reasoning:" tag is found, use all text after the score as reasoning
            score_index = response_text.lower().find('score')
            reasoning = response_text[score_index:].strip() if score_index != -1 else response_text

        reasoning = reasoning.strip('"')
        return EvaluationResult(score=score, reasoning=reasoning)


class AggregatedEvaluationResult(BaseModel):
    overall_score: float
    aspect_scores: Dict[str, int]
    combined_reasoning: str


class MultiEvaluator:
    def __init__(self,
                 evaluators: List[OpenAIEvaluator]):
        self.evaluators = evaluators

    async def evaluate(self,
                       system_prompt: str,
                       user_input: str,
                       current_output: str,
                       previous_output: Optional[str] = None) -> AggregatedEvaluationResult:
        evaluation_tasks = [
            evaluator.evaluate(system_prompt, user_input, current_output, previous_output)
            for evaluator in self.evaluators
        ]
        results = await asyncio.gather(*evaluation_tasks)

        aspect_scores = {evaluator.evaluation_aspect: result.score for evaluator, result in
                         zip(self.evaluators, results)}
        overall_score = mean(aspect_scores.values())
        combined_reasoning = "\n".join(
            f"{evaluator.evaluation_aspect}: {result.reasoning}" for evaluator, result in zip(self.evaluators, results))

        return AggregatedEvaluationResult(
            overall_score=overall_score,
            aspect_scores=aspect_scores,
            combined_reasoning=combined_reasoning
        )
