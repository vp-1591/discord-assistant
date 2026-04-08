---
description: Evaluate and iterate on a draft prompt against a scenario rubric
---

# Prompt Evaluation Workflow

1. **Read Scenario**: Read the target JSON configuration file (e.g., `scenarios/test_x.json`) to understand the constraints and rubrics.
2. **Setup Draft Prompt**: Extract the target prompt from `src/config/prompts.py` and write it to `tests/test_factory/.drafts/draft_prompt.txt`. 
3. **Execute Harness**: Run `python tests/test_factory/run_harness.py --scenario scenarios/test_x.json --prompt tests/test_factory/.drafts/draft_prompt.txt`. Wait for completion up to 3 minutes
4. **Analyze Output**: Read `trace_output.json` and compare the exact outputs against the rubrics from the scenario setup. If output failed rubrics because of part of prompt other than in draft_prompt.txt, immediately interrupt this workflow and notify me in the chat with proposed fix
5. **Iterate Draft**: If any rubric fails, use /prompt-engineering skill to edit `tests/test_factory/.drafts/draft_prompt.txt` to fix the failure.
6. **Repeat & Finalize**: Loop steps 3-5 up to a maximum of 3 times. Once successful, propose the final prompt changes in chat. Do not edit prompts.py!