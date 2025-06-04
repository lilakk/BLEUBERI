#! /bin/bash

cd eval/arena-hard
python3 show_result.py --style-control --load-bootstrap --load-battles
cd ../..

# for arena-hard-v2.0, there can be small floating point changes in the results compared to what's reported in the paper
cd eval/arena-hard-v2.0
python3 show_result.py --judge-names gpt-4.1-mini --control-features markdown length
cd ../..

cd eval/FastChat/fastchat/llm_judge
python show_result.py
cd ../../../..

cd eval/WildBench
bash leaderboard/show_eval.sh score_only
cd ../..
