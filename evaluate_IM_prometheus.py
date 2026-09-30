import argparse
from prometheus_eval import PrometheusEval
from prometheus_eval.vllm import VLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
import pandas as pd
from tqdm import tqdm
import os

import random
random.seed(2025)

parser = argparse.ArgumentParser(description='get_gens')
parser.add_argument('--gpu', type=str, default='1', help='which gpu id to use')
parser.add_argument('--gen_path', type=str)
parser.add_argument('--test_path', type=str)
args = parser.parse_args()
gpu_id = args.gpu.strip()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")

faithfulness_rubric = {
    "criteria": "Assuming the explanation does not contradict the utterance, what is the depth and strategic utility of the model's internal reasoning within this counseling context?",

    "score1_description": "Hallucination / Contradiction: The explanation is factually incorrect. It describes content, strategies, or intents strictly ABSENT from or contradicted by the Utterance (e.g., claiming to ask a probing question when the utterance is just a passive affirmation). *Note: Task relevant text like 'this is the last utterance' do not belong here.*",

    "score2_description": "Generic / Filler: The explanation is accurate but provides zero task-specific insight. These are statements not related to counseling task (e.g., 'Please let me know how you would like to proceed', 'I am waiting for the next utterance.').",

    "score3_description": "Contextual / Persona-Maintenance: The explanation references the specific context or rules but does not analyze the current turn. Includes citing the persona, acknowledging the system state, or extending the dialogue functionally (e.g., 'I will respond as the counselor', 'This is the last response according to the...', 'Remember to validate', 'I am taking notes'). It is aware of the context, but lacks analysis of the specific action.",

    "score4_description": "Descriptive Analysis (The 'What'): Accurately describes the specific tactical action taken in the current utterance. It summarizes *what* the model did or what specific topic it addressed (e.g., 'The goal of the conversation is...', 'I asked an open-ended question about their sleep schedule'). It captures the action but lacks the deeper strategic 'why'.",

    "score5_description": "Causal Intent / Strategic Reasoning (The 'Why'): Explains the specific reasoning  or intent behind the utterance. It links the utterance to a goal, a hypothesis about the user, or a specific strategic intent (e.g., 'Note: I'm trying to steer the conversation towards', 'I'm trying to get to the root of the issue', 'This is a gentle nudge')."
}

def evaluate_causality(conversation,explanation, utterance, model=model):
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)
    instruction = f"""You are an expert evaluator assessing the internal reasoning of an AI counseling assistant.

Your task is to evaluate the quality, depth, and accuracy of the AI's internal explanation (provided as the response to be graded) regarding the utterance it just generated.

Use the following context to make your judgment:

### Conversation Context:
{conversation}

### AI's Actual Utterance:
{utterance}

Read the conversation context and the AI's actual utterance for reference only. Then, evaluate the provided explanation based on the grading rubric to determine if its relevant to the task and utterance.
"""
    feedback, score = judge.single_absolute_grade(
        instruction=instruction,
        response=explanation,
        rubric=SCORE_RUBRIC_TEMPLATE.format(**faithfulness_rubric),
    )
    return score, feedback

df = pd.read_csv(args.gen_path+".csv")
if 'cactus' not in args.test_path:
    test_cactus = pd.read_json(args.test_path, orient='records', lines=True)

else:    
    test_cactus = pd.read_json(args.test_path)
    
df['conversation_context'] = test_cactus['input']
print(f"Len of df before: ", len(df))
df0 = df.dropna(subset=['comment'], inplace=False)

print(f"Len of df after dropping no comments: ", len(df0))

scores = []
feedbacks = []
for i,row in tqdm(df0.iterrows(), total=len(df0)):
    score, feedback = evaluate_causality(row['conversation_context'], row['comment'], row['full_text'])
    scores.append(score)
    feedbacks.append(feedback)

df0['score'] = scores
df0['feedback'] = feedbacks

df0.to_csv(args.gen_path+".csv", index=False)
print(args.gen_path)
print(df0['score'].value_counts())