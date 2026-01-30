from aenum import extend_enum
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetric,
    SamplingMethod,
)
from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
)
import numpy as np


MATH_QUERY_TEMPLATE = """
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip()


def math_500_prompt(line, task_name: str = None):
    query = MATH_QUERY_TEMPLATE.format(prompt=line["problem"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"ANSWER: {line['solution']}"],
        gold_index=0,
    )


def record_to_sample(record):
    query = record["problem"]
    target = record["answer"]
    return Sample(input=query, target=target)


latex_gold_match = SampleLevelMetric(
    metric_name="latex_gold_match",
    sample_level_fn=MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        fallback_mode="first_match",
        precision=5,
        gold_extraction_target=(LatexExtractionConfig(),),
        # Match boxed first before trying other regexes
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
        aggregation_function=max,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

extend_enum(Metrics, "latex_gold_match", latex_gold_match)

math500_custom = LightevalTaskConfig(
    name="math500_custom",
    prompt_function=math_500_prompt,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.latex_gold_match],
    version=2,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=model_graded_fact(),
)

math500_passk_custom = LightevalTaskConfig(
    name="math500_passk_custom",
    prompt_function=math_500_prompt,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
    ],
    version=2,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=model_graded_fact(),
)

math500_avgn_custom = LightevalTaskConfig(
    name="math500_avgn_custom",
    prompt_function=math_500_prompt,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        Metrics.avg_at_n_math(sample_params={"n": 16}),
    ],
    version=2,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=model_graded_fact(),
)

TASKS_TABLE = [
    math500_custom,
    math500_passk_custom,
    math500_avgn_custom,
]