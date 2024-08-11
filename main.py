import asyncio
import logging
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from config import config
from evaluator import OpenAIEvaluator, MultiEvaluator
from reviser import Reviser, RevisionInput
from task_writer_test_input import TASK_WRITER_SYSTEM_PROMPT, TASK_WRITER_INITIAL_INPUT, TASK_WRITER_INITIAL_OUTPUT
from tracing import tracer
from output_handler import write_output_files


logging.basicConfig(level=config['logging']['level'], format=config['logging']['format'])
logger = logging.getLogger(__name__)


def get_llm(config: dict, model_type: str):
    model_config = config['llm'][model_type]
    provider = model_config['provider']

    if provider == 'openai':
        return ChatOpenAI(model=model_config['name'], api_key=config['env']['OPENAI_API_KEY'])
    elif provider == 'anthropic':
        return ChatAnthropic(model=model_config['name'], api_key=config['env']['ANTHROPIC_API_KEY'])
    else:
        raise ValueError(f"Unsupported model provider: {provider}")


@tracer(run_type="chain", name="Main Revision Pipeline")
async def main():
    agent_llm = get_llm(config, 'agent_model')
    reviser_llm = get_llm(config, 'reviser_model')

    evaluators = [
        OpenAIEvaluator(
            api_key=config['env']['OPENAI_API_KEY'],
            model=config['llm']['evaluator_model']['name'],
            evaluation_aspect=aspect
        )
        for aspect in config['evaluation']['aspects']
    ]
    multi_evaluator = MultiEvaluator(evaluators)
    reviser = Reviser(agent_llm, reviser_llm, multi_evaluator)

    revision_input = RevisionInput(
        system_prompt=TASK_WRITER_SYSTEM_PROMPT,
        user_input=TASK_WRITER_INITIAL_INPUT,
        initial_output=TASK_WRITER_INITIAL_OUTPUT
    )

    result = await reviser.revise(revision_input)

    logger.info("Final Revised Output:")
    logger.info(result.final_output)

    logger.info("Revision History:")
    for i, (revision, evaluation) in enumerate(zip(result.revision_history, result.evaluation_history)):
        logger.info(f"Revision {i}:")
        logger.info(revision)
        logger.info(f"Overall Evaluation Score: {evaluation.overall_score}")
        logger.info(f"Aspect Scores: {evaluation.aspect_scores}")
        logger.info(f"Combined Reasoning: {evaluation.combined_reasoning}")

    write_output_files(result.final_output, result.history_log, debug=config.get('DEBUG', False))


if __name__ == "__main__":
    asyncio.run(main())
