import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from config import config
from prompts import FEEDBACK_PROMPT, REVISION_PROMPT
from tracing import tracer

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


class Reviser:
    def __init__(self, agent_llm, reviser_llm, max_iterations: int = config['reviser']['max_iterations']):
        self.agent_llm = agent_llm
        self.reviser_llm = reviser_llm
        self.max_iterations = max_iterations

        self.feedback_chain = FEEDBACK_PROMPT | self.reviser_llm | StrOutputParser()
        self.revision_chain = REVISION_PROMPT | self.agent_llm | StrOutputParser()

    @tracer(run_type="chain", name="Revision Process")
    def revise(self, revision_input: RevisionInput) -> RevisionResult:
        current_output = revision_input.initial_output or "No initial output provided."
        revision_history = [current_output]

        for i in range(self.max_iterations):
            try:
                logger.info(f"Starting revision iteration {i + 1}")

                feedback = self.feedback_chain.invoke({
                    "system_prompt": revision_input.system_prompt,
                    "user_input": revision_input.user_input,
                    "current_output": current_output
                })

                current_output = self.revision_chain.invoke({
                    "system_prompt": revision_input.system_prompt,
                    "user_input": revision_input.user_input,
                    "current_output": current_output,
                    "feedback": feedback
                })

                revision_history.append(current_output)
                logger.info(f"Revision iteration {i + 1} complete")

            except Exception as e:
                logger.error(f"Error during revision iteration {i + 1}: {str(e)}")
                break

        return RevisionResult(final_output=current_output, revision_history=revision_history)


def get_llm(model_config: dict, api_key: str):
    if model_config['provider'] == 'openai':
        return ChatOpenAI(model=model_config['name'], api_key=api_key)
    elif model_config['provider'] == 'anthropic':
        return ChatAnthropic(model=model_config['name'], api_key=api_key)
    else:
        raise ValueError(f"Unsupported model provider: {model_config['provider']}")


@tracer(run_type="chain", name="Main Revision Pipeline")
def main():
    agent_model_config = config['llm']['agent_model']
    reviser_model_config = config['llm']['reviser_model']

    agent_llm = get_llm(agent_model_config, config['env']['OPENAI_API_KEY'])
    reviser_llm = get_llm(reviser_model_config, config['env']['ANTHROPIC_API_KEY'])

    reviser = Reviser(agent_llm, reviser_llm)

    revision_input = RevisionInput(
        system_prompt="<agent's system prompt>",
        user_input="<user input>",
        initial_output="<agent's output>",
    )

    result = reviser.revise(revision_input)

    logger.info("Final Revised Output:")
    logger.info(result.final_output)

    logger.info("Revision History:")
    for i, revision in enumerate(result.revision_history):
        logger.info(f"Revision {i}:")
        logger.info(revision)


if __name__ == "__main__":
    main()