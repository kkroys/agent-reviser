from langchain_core.prompts import ChatPromptTemplate


FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You're a professional content reviewer. Provide constructive feedback based on the given context and requirements. Evaluate the current output, suggest improvements, and highlight any missing or superfluous elements."),
    ("human", "Context and Requirements: {system_prompt}\n\nUser Input: {user_input}\n\nCurrent Output: {current_output}\n\nPlease provide your feedback and suggestions for improvement:"),
])


REVISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "User Input: {user_input}\n\nCurrent Output: {current_output}\n\nFeedback: {feedback}\n\nPlease provide an improved version of the output:"),
])