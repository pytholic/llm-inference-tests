import json
import os
import config
import requests
from dataclasses import dataclass
from typing import Optional
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
avg_metrics = {
    "total_duration": 0,
    "time_to_first_token": 0,
    "time_per_output_token": 0,
    "tokens_per_second": 0,
    "eval_count": 0,
    "eval_duration": 0,
}


@dataclass
class LLMOptions:
    temperature: Optional[float] = None
    num_ctx: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    num_keep: Optional[int] = None
    system: Optional[str] = None
    num_gpu: Optional[int] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def generate_response(
    prompt: str,
    model: str = config.MODEL_NAME,
    host: str = config.HOST,
    port: int = config.PORT,
    options: LLMOptions | None = None,
):
    url = f"http://{host}:{port}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}

    options_dict = options.to_dict() if options else None

    if options_dict:
        payload["options"] = options_dict
    if options.system:
        payload["system"] = options.system

    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line.decode("utf-8"))
                    if json_response.get("done"):
                        # Capture final metrics
                        metrics = {
                            "total_duration": json_response.get("total_duration", 0)
                            / 1e9,  # Convert to seconds
                            "load_duration": json_response.get("load_duration", 0) / 1e9,
                            "prompt_eval_count": json_response.get("prompt_eval_count", 0),
                            "prompt_eval_duration": json_response.get("prompt_eval_duration", 0)
                            / 1e9,
                            "eval_count": json_response.get("eval_count", 0),
                            "eval_duration": json_response.get("eval_duration", 0) / 1e9,
                        }
                        break
                    print(json_response.get("response", ""), end="", flush=True)

        print("\n\nPerformance Metrics:")
        print(f"Total Duration: {metrics['total_duration']:.3f}s")
        print(f"Load Duration: {metrics['load_duration']:.3f}s")
        print(f"Prompt Eval Count: {metrics['prompt_eval_count']}")
        print(f"Prompt Eval Duration: {metrics['prompt_eval_duration']:.3f}s")
        print(f"Eval Count: {metrics['eval_count']}")
        print(f"Eval Duration: {metrics['eval_duration']:.3f}s")
        print(f"Tokens/second: {metrics['eval_count'] / metrics['eval_duration']:.2f}")

        global avg_metrics
        avg_metrics["total_duration"] += metrics["total_duration"]
        avg_metrics["time_to_first_token"] += metrics["prompt_eval_duration"]
        avg_metrics["time_per_output_token"] += metrics["eval_duration"] / metrics["eval_count"]
        avg_metrics["tokens_per_second"] += metrics["eval_count"] / metrics["eval_duration"]
        avg_metrics["eval_count"] += metrics["eval_count"]
        avg_metrics["eval_duration"] += metrics["eval_duration"]
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # prompt = "Write a poem about programming."
    prompt = """
    # Report Structure

    The report must contain two main sections: Findings and Impression.

    ## Findings

    Organize the Findings section into the following five subsections:
    1. Quality of Exam
    2. Medical Devices and Foreign Bodies
    3. Lungs/Pleura/Diaphragms
    4. Cardiomediastinum and Hila
    5. Other Body Parts (Bones, Soft Tissue, Abdomen)

    For each subsection:
    - Positive Findings: Describe all relevant findings from the JSON data.
        - Grouping: Group similar or related findings and describe them together. For example, if costophrenic angle blunting and pleural effusion are positive on the same side, describe them together.
    - Negation: If there are no notable abnormalities, state this clearly (e.g., "Normal cardiomediastinum and hila.").
    - Physiological Consistency: When interpreting the JSON data and writing the report, ensure that all described findings are physiologically consistent and make sense in the context of established medical knowledge. Cross-verify that the direction of any shifts (e.g., mediastinal or tracheal) logically corresponds with the location and severity of findings like pleural effusions or pneumothorax. If inconsistencies arise, interpret the findings in a way that aligns with normal physiological responses.
    - Do NOT write clinical diagnosis such as pneumonia or pulmonary edema in this Findings section. Interpretations will be included in the Impression section.

    ## Impression

    In the Impression section:
    - Summarize ONLY the most important findings, such as clinical diagnoses or observations that may require further treatment.
    - For each key finding, include one main piece of supporting evidence from the Findings section.
    - Write clearly and concisely.
    - Do not recommend any further evaluation or treatment.

    # Interpreting Severity Levels in JSON Data

    The JSON input includes severity values represented by numbers (or words): 1 (or borderline/minimal), 2 (or moderate), 3 (or severe)

    For Findings, interpret them following the next:
    - 1 (or borderline/minimal): possible/borderline/mild/minimal/subtle
    - 2 (or moderate): no need to mention or moderate
    - 3 (or severe): severe/large/no need to mention

    For clinical diagnosis in Impressions, interpret them following the next:
    - 1 (or borderline/minimal): possible or mild
    - 2 (or moderate): suspected or probable
    - 3 (or severe): consistent with

    Use appropriate terminology to express the severity levels in your report, ensuring they accurately reflect the context of each finding. Do not mention the severity level number directly in the sentences.

    # Suggesting Clinical Diagnosis

    When suggesting clinical diagnoses:
    - Use the severity as a measure of certainty.
    - If two clinical diagnoses indicate the same radiologic feature, treat them as differential diagnoses.
        - For example, if pneumonia has a severity of 1 (or borderline/minimal) and pulmonary edema has a severity of 2 (or moderate), state that pulmonary edema is more likely than pneumonia.

    # Additional Rules

    - Be concise as much as possible
    - Use CVC for central venous catheter.
    - Refer to PICC or peripheral line as peripherally inserted venous catheter.
    - Describe "cardiomegaly" as a "widened cardiac silhouette."
    - Avoid mentioning nasogastric tube visibility.
    - Omit CPA blunting and volume loss if they are evident from major findings like large pleural effusion or total lung collapse.

    # Task

    Now, write a radiologic report based on the following JSON input:

    {'mediastinal_mass': {'is_abnormal': True, 'mediastinal_mass_side_location': ['Right upper'], 'mediastinal_mass_severity': 'moderate'}, 'subsegmental_atelectasis': {'is_abnormal': True, 'subsegmental_atelectasis_location': ['Left lung lower zone']}, 'pulmonary_mass_nodule': {'is_abnormal': True, 'pulmonary_mass_nodule_multiplicity': 'Single', 'pulmonary_mass_nodule_location': ['Right lung lower zone'], 'pulmonary_mass_nodule_severity': 'Small'}, 'calcified_nodule': {'is_abnormal': True, 'calcified_nodule_multiplicity': 'Single', 'calcified_nodule_location': ['Right lung lower zone'], 'calcified_nodule_severity': ['Small']}, 'malignancy': {'is_abnormal': True, 'malignancy_severity': 'borderline/minimal'}}
    """

    options = LLMOptions(
        temperature=0,
        num_ctx=2048,
        top_k=1,
        top_p=1.0,
        num_predict=512,
        num_keep=0,
        seed=42,
        system="You are an expert radiologist tasked with writing a radiologic report for a chest X-ray based on the provided JSON-formatted information. Your report should be professional and written in natural language.",
        num_gpu=1,
    )

    # Generate response
    num_runs = 5
    for i in range(num_runs):
        print(f"\n***Response {i+1} from model***\n")
        generate_response(prompt, options=options)
        time.sleep(0.5)

    print("\n\nAverage Metrics:")
    print(f"Total Duration: {avg_metrics['total_duration'] / num_runs:.3f}s")
    print(f"Time to First Token: {avg_metrics['time_to_first_token'] / num_runs:.3f}s")
    print(f"Time Per Output Token: {avg_metrics['time_per_output_token'] / num_runs:.3f}s")
    print(f"Tokens Per Second: {avg_metrics['tokens_per_second'] / num_runs:.2f}")
    print(f"Eval Count: {avg_metrics['eval_count'] / num_runs}")
    print(f"Eval Duration: {avg_metrics['eval_duration'] / num_runs:.3f}s")
