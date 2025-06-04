"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

import numpy as np
from tqdm import tqdm

from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)


def load_existing_judgments(output_file_path, mode):
    """Loads existing judgments from the output file."""
    existing_judgments = set()
    if not os.path.exists(output_file_path):
        print(f"No existing judgment file found at: {output_file_path}")
        return existing_judgments

    print(f"Loading existing judgments from: {output_file_path}")
    count = 0
    malformed_count = 0
    with open(output_file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                question_id = data.get("question_id")

                if question_id is None: # Skip if no question_id
                    malformed_count +=1
                    continue

                if mode == "single":
                    # The judgment output for single mode has "model_id"
                    # Changed to "model" based on user's file format
                    model_key_to_check = "model" 
                    model_id = data.get(model_key_to_check) # Use the new variable
                    if model_id is not None:
                        existing_judgments.add((question_id, model_id))
                        count += 1
                    else:
                        malformed_count += 1
                        # print(f"Debug: Malformed single - QID: {question_id}, Keys: {data.keys()}, Expected key: {model_key_to_check}")
                else:  # pairwise-baseline or pairwise-all
                    # The judgment output for pair mode has "model_1" and "model_2"
                    model_1 = data.get("model_1")
                    model_2 = data.get("model_2")
                    if model_1 is not None and model_2 is not None:
                        # The order in MatchPair is (model1, model2)
                        # and this order is preserved from make_match / make_match_all_pairs
                        existing_judgments.add((question_id, model_1, model_2))
                        count += 1
                    else:
                        malformed_count +=1
            except json.JSONDecodeError:
                malformed_count += 1
                # print(f"Warning: Skipping malformed JSON line in {output_file_path}: {line.strip()}") # Can be noisy
    
    if malformed_count > 0:
        print(f"Warning: Skipped {malformed_count} malformed or incomplete lines in existing judgments.")
    print(f"Successfully loaded {count} unique existing judgments.")
    return existing_judgments


def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            # Ensure answers exist
            if q_id not in model_answers.get(m_1, {}) or \
               q_id not in model_answers.get(baseline_model, {}):
                # print(f"Warning: Skipping Q:{q_id} for {m_1} vs {m_2} due to missing answer(s).")
                continue

            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            
            current_ref = None
            if ref_answers is not None and judge.ref_based:
                if judge.model_name in ref_answers and q_id in ref_answers[judge.model_name]:
                    current_ref = ref_answers[judge.model_name][q_id]
                # else:
                    # print(f"Warning: No ref answer for Q:{q_id} with judge {judge.model_name} though ref_based.")


            match = MatchPair(
                dict(q),
                m_1,
                m_2,
                a_1,
                a_2,
                judge,
                ref_answer=current_ref,
                multi_turn=multi_turn,
            )
            matches.append(match)
    return matches


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None, # Not used but kept for signature consistency if called polymorphically
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q["question_id"]
                m_1 = models[i]
                m_2 = models[j]

                if q_id not in model_answers.get(m_1, {}) or \
                   q_id not in model_answers.get(m_2, {}):
                    # print(f"Warning: Skipping Q:{q_id} for {m_1} vs {m_2} due to missing answer(s).")
                    continue
                
                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]

                current_ref = None
                if ref_answers is not None and judge.ref_based:
                    if judge.model_name in ref_answers and q_id in ref_answers[judge.model_name]:
                        current_ref = ref_answers[judge.model_name][q_id]
                    # else:
                        # print(f"Warning: No ref answer for Q:{q_id} with judge {judge.model_name} though ref_based.")
                
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=current_ref,
                    multi_turn=multi_turn,
                )
                matches.append(match)
    return matches


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None, # Not used but kept for signature consistency
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]

            if q_id not in model_answers.get(m, {}):
                # print(f"Warning: Skipping Q:{q_id} for model {m} due to missing answer.")
                continue
            a = model_answers[m][q_id]

            current_ref = None
            if ref_answers is not None and judge.ref_based:
                if judge.model_name in ref_answers and q_id in ref_answers[judge.model_name]:
                    current_ref = ref_answers[judge.model_name][q_id]
                # else:
                    # print(f"Warning: No ref answer for Q:{q_id} with judge {judge.model_name} though ref_based.")
            
            matches.append(
                MatchSingle(
                    dict(q), m, a, judge, ref_answer=current_ref, multi_turn=multi_turn
                )
            )
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"
    ref_answer_dir = f"data/{args.bench_name}/reference_answer"

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    # Handle cases where ref_answers might not exist or be relevant for all judges
    try:
        ref_answers = load_model_answers(ref_answer_dir)
    except FileNotFoundError:
        print(f"Reference answer directory not found: {ref_answer_dir}. Proceeding without reference answers.")
        ref_answers = {} # Ensure ref_answers is a dict

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    # Define output_file and play_a_match_func based on mode
    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
        make_match_func = make_match_single
        baseline_model = None
    else:  # pairwise modes
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        # Ensure consistent output file naming for pairwise modes
        pair_mode_suffix = "pair_all" if args.mode == "pairwise-all" else "pair_baseline"
        output_file = (
             f"data/{args.bench_name}/model_judgment/{args.judge_model}_{pair_mode_suffix}.jsonl"
        )
        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:  # pairwise-baseline
            make_match_func = make_match
            baseline_model = args.baseline_model
            if baseline_model not in models: # If baseline is also in model-list, it's fine, handled by m1==m2 check
                # if baseline model is not in general answer dir, it might cause issues later
                if baseline_model not in model_answers:
                     print(f"Warning: Baseline model {baseline_model} answers not found in {answer_dir}. "
                           "Ensure it exists or it will be skipped in pairings.")


    # Create directory for output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Output file will be: {output_file}")

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make all potential matches
    all_matches = []
    # For default questions (non-reference based typically, unless judge is ref_based)
    all_matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model, ref_answers=ref_answers
    )
    # For math questions (reference based typically)
    all_matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers=ref_answers, # Pass ref_answers here
    )
    # For multi-turn default questions
    all_matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        ref_answers=ref_answers,
        multi_turn=True,
    )
    # For multi-turn math questions
    all_matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers=ref_answers, # Pass ref_answers here
        multi_turn=True,
    )
    
    # ---- MODIFICATION: Load existing judgments and filter matches ----
    existing_judgments = load_existing_judgments(output_file, args.mode)
    
    matches_to_play = []
    skipped_count = 0
    if not all_matches:
        print("No matches were generated based on the inputs. Check model answers and question categories.")
    else:
        for match_item in all_matches:
            qid = match_item.question["question_id"]
            identifier = None
            if args.mode == "single":
                # MatchSingle has .model attribute
                identifier = (qid, match_item.model)
            else:
                # MatchPair has .model1 and .model2 attributes
                identifier = (qid, match_item.model1, match_item.model2)
            
            if identifier and identifier in existing_judgments:
                skipped_count += 1
                continue
            matches_to_play.append(match_item)

    print(f"Total potential matches generated: {len(all_matches)}")
    if existing_judgments: # Check if the set is not empty before printing its length
        print(f"Found {len(existing_judgments)} existing judgment records.")
    print(f"Number of new matches to play: {len(matches_to_play)}")
    print(f"Number of matches skipped (already judged): {skipped_count}")
    # ---- END MODIFICATION ----

    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode
    match_stat["judge_model"] = args.judge_model # Changed from "judge"
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_potential_matches"] = len(all_matches) # Original total
    match_stat["num_matches_to_play"] = len(matches_to_play) # New stat
    match_stat["num_already_judged"] = skipped_count        # New stat
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("\nStats:")
    print(json.dumps(match_stat, indent=4))
    
    if not matches_to_play:
        print("\nNo new matches to play. Exiting.")
    else:
        print(f"\nProceeding to judge {len(matches_to_play)} new matches.")

        # Play matches
        if args.parallel == 1:
            for match_item_loop in tqdm(matches_to_play, desc="Playing matches"):
                play_a_match_func(match_item_loop, output_file=output_file)
        else:
            # Inner function for parallel execution
            def play_a_match_wrapper(match_wrapper_item):
                play_a_match_func(match_wrapper_item, output_file=output_file)

            np.random.seed(0) # Seed for reproducible shuffling
            shuffled_matches_to_play = list(matches_to_play) # Make a mutable copy for shuffle
            np.random.shuffle(shuffled_matches_to_play)

            with ThreadPoolExecutor(args.parallel) as executor:
                list(tqdm( # Consume the iterator to ensure execution and show progress
                    executor.map(play_a_match_wrapper, shuffled_matches_to_play),
                    total=len(shuffled_matches_to_play),
                    desc="Playing matches (parallel)"
                ))
        print(f"\nFinished judging. Results saved to {output_file}")
