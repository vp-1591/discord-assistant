---
description: Evaluate and iterate on a draft prompt against a scenario rubric
---

# Prompt Evaluation Workflow

1. **Read Scenario**: Read the target JSON configuration file (e.g., `scenarios/test_x.json`) to understand the constraints and rubrics.
2. **Setup Draft Prompt**: Extract the target prompt from `src/config/prompts.py` and write it to `tests/test_factory/.drafts/draft_prompt.txt`. Use "Глава Архива" in place of bot_name.
3. **Execute Harness**: Run `python tests/test_factory/run_harness.py --scenario scenarios/test_x.json --prompt tests/test_factory/.drafts/draft_prompt.txt --iteration N` where `N` is the current iteration number (start at 1). Running with `--iteration 1` automatically purges all previous `trace_iter_*.json` files. Wait for completion up to 3 minutes.
4. **Analyze Output**: Read `tests/test_factory/trace_iter_N.json` (matching the current iteration) and compare the exact outputs against the rubrics from the scenario setup. If output failed rubrics because of part of prompt other than in draft_prompt.txt, immediately interrupt this workflow and notify me in the chat with proposed fix. If output failed because of harness error, you must notify me in the chat and you MUST NOT change code.
5. **Iterate Draft**: If any rubric fails, use /prompt-engineering skill to edit `tests/test_factory/.drafts/draft_prompt.txt` to fix the failure.
6. **Repeat & Finalize**: Loop steps 3-5 up to a maximum of 3 times, incrementing `N` on each loop (so you produce `trace_iter_1.json`, `trace_iter_2.json`, `trace_iter_3.json` for easy comparison). Once successful, propose the final prompt changes in chat. Do not edit prompts.py!