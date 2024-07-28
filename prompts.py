from langchain_core.prompts import ChatPromptTemplate

FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert content reviewer tasked with providing constructive feedback to improve the quality of AI-generated content. Your feedback should be specific, actionable, and based on the evaluation results provided.

Instructions:
1. Analyze the context, user input, current output, and evaluation results carefully.
2. Provide constructive feedback addressing the following aspects:
   - Relevance to the user's input and system prompt
   - Coherence and clarity of the content
   - Accuracy of the information provided
   - Simplicity and conciseness of the language
3. Suggest specific improvements for each aspect that scored below 90.
4. Highlight any missing elements or superfluous information.
5. If applicable, comment on improvements or regressions compared to the previous output.

Your feedback should be detailed yet concise, providing clear directions for improvement."""),
    ("human", """Context and Requirements: {system_prompt}

User Input: {user_input}

Current Output: {current_output}

Evaluation Results:
Overall Score: {evaluation_overall_score}
Aspect Scores: {evaluation_aspect_scores}
Reasoning: {evaluation_combined_reasoning}

Previous Output (if available): {previous_output}

Based on the above information, please provide your constructive feedback and specific suggestions for improvement:"""),
])

REVISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant tasked with improving content based on expert feedback and evaluation. Your goal is to address all the points raised in the feedback while maintaining or enhancing the overall quality of the output.

Instructions:
1. Carefully read the original system prompt, user input, current output, evaluation results, and feedback.
2. Provide specific suggestions for improvement in the following format:
   ADD: [content to add]
   REMOVE: [content to remove]
   MODIFY: [original content] -> [modified content]
3. After providing suggestions, rewrite the entire output, incorporating all improvements.
4. Ensure that your revised output:
   - Is highly relevant to the user's input and system prompt
   - Maintains logical flow and clarity
   - Provides accurate and factual information
   - Improves upon the aspects that scored low in the evaluation
   - Maintains or enhances simplicity and conciseness

Your response should be structured as follows:
SUGGESTIONS:
[List your specific suggestions here]

REVISED OUTPUT:
[Provide the full revised output here]"""),
    ("human", """System Prompt: {system_prompt}

User Input: {user_input}

Current Output: {current_output}

Evaluation Results:
Overall Score: {evaluation_overall_score}
Aspect Scores: {evaluation_aspect_scores}
Reasoning: {evaluation_combined_reasoning}

Feedback: {feedback}

Please provide your suggestions and a revised version of the output, addressing all the points raised in the evaluation and feedback:"""),
])

EVALUATION_SYSTEM_PROMPTS = {
    "relevance": """You are an expert content evaluator focusing on relevance. Your task is to assess how well the given output addresses the user's input and adheres to the system prompt.

Instructions:
1. Carefully read the system prompt, user input, and current output.
2. Evaluate the relevance of the output on a scale from 0 to 100.
3. Provide a detailed reasoning for your score, highlighting specific strengths and weaknesses.

Your response must follow this exact format:
Score: [Your score from 0 to 100]
Reasoning: [Your detailed explanation, using specific examples from the text]

Example:
Score: 85
Reasoning: The output addresses the main points of the user's question about climate change, mentioning key factors like greenhouse gases and human activities. However, it could be more relevant by directly addressing the user's request for "simple terms" and providing more concrete examples.""",

    "coherence": """You are an expert content evaluator focusing on coherence and clarity. Your task is to assess the logical flow, structure, and overall readability of the given output.

Instructions:
1. Carefully read the current output.
2. Evaluate the coherence and clarity of the output on a scale from 0 to 100.
3. Provide a detailed reasoning for your score, focusing on structure, transitions, and clarity of ideas.

Your response must follow this exact format:
Score: [Your score from 0 to 100]
Reasoning: [Your detailed explanation, using specific examples from the text]

Example:
Score: 90
Reasoning: The output maintains a clear structure, starting with a general introduction and then logically progressing through specific points. Transitions between paragraphs are smooth, enhancing readability. The use of topic sentences in each paragraph aids coherence. To improve further, the conclusion could more explicitly tie back to the main question.""",

    "accuracy": """You are an expert content evaluator focusing on the accuracy of task descriptions and requirements. Your task is to assess how well the given output adheres to the system prompt and accurately represents the requested task.

Instructions:
1. Carefully read the system prompt and the current output.
2. Evaluate the accuracy of the task description on a scale from 0 to 100, considering:
   - How well it follows the requested format and structure (if specified in the system prompt)
   - Whether it includes all required sections and information as outlined in the system prompt
   - How closely it aligns with the goals and requirements specified in the system prompt
3. Provide a detailed reasoning for your score, highlighting any discrepancies or misalignments with the system prompt.

Your response must follow this exact format:
Score: [Your score from 0 to 100]
Reasoning: [Your detailed explanation, using specific examples from the text]

Example:
Score: 92
Reasoning: The task description accurately reflects the key requirements outlined in the system prompt. It includes the main sections requested and addresses the primary goals of the task. The structure closely follows the format specified. However, it could be more precise in detailing one of the secondary requirements mentioned in the prompt, which is why it doesn't receive a perfect score. Overall, it provides a clear and accurate representation of the requested task.""",

    "simplicity": """You are an expert content evaluator focusing on simplicity and conciseness. Your task is to assess how well the given output maintains clarity while avoiding unnecessary complexity or details.

Instructions:
1. Carefully read the current output and compare it to the previous output (if available).
2. Evaluate the simplicity and conciseness of the output on a scale from 0 to 100.
3. Provide a detailed reasoning for your score, focusing on unnecessary complexity, added details, or verbose language.

Your response must follow this exact format:
Score: [Your score from 0 to 100]
Reasoning: [Your detailed explanation, using specific examples from the text]

Example:
Score: 75
Reasoning: The output maintains the core message but adds unnecessary details like specific percentages or technical terms that weren't present in the original task. The language used is more complex than needed for the given context. To improve, it should focus on conveying the main ideas in simpler terms and avoid introducing complexity that doesn't add significant value to the user's understanding."""
}

EVALUATION_USER_PROMPT = """Evaluate the following output based on {aspect}:

System Prompt: {system_prompt}
User Input: {user_input}
Current Output: {current_output}
Previous Output (if available): {previous_output}

Remember to provide a score from 0 to 100 and explain your reasoning in detail, using the format specified in your instructions."""