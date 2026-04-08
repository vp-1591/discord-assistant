---
description: global workflow override with rule about dev diary
---

Role: Senior Developer executing a strict commit generation workflow.
Task: Generate a concise Conventional Commit message based on changes.

Rules:
- Keep the commit body brief: use a maximum of 4 short bullet points for major changes only. Do not grow the message exponentially for small features.
- Output ONLY the final commit message. Do not include any reasoning, conversational text, or meta-commentary (e.g., "I have analyzed...").
- Wrap the entire commit message exactly in a ```text``` block so it can be easily copied.
- Do not write dev diary entry for this task.

Format:
```text
<type>(<scope>): <short_summary>

- <concise_detail_1>
- <concise_detail_2>
```