# Implementation Plan: Social Memory & Auditor System

## 1. Core Objective

Transform the bot's memory from a simple chat summary into a "Social Memory" system. The bot must track subjective relationships with users in `opinions.json`, using an "Auditor" agent (Agent 3) to process interactions asynchronously and a "Entity Recognizer" (Agent 1.5) to enable multi-user context.

## 2. Data Structure: `opinions.json`

Entries are keyed by Discord Snowflake IDs. Store only descriptive strings.

```
{
  "DISCORD_ID": {
    "name": "CurrentNickname",
    "head_of_archive_stance": "3-sentence max description of the bot's current feeling toward this person.",
    "interaction_history": "3-sentence max summary of recent impactful exchanges."
  }
}
```

## 3. The Multi-Agent Pipeline

### Phase A: Pre-Processing (Asynchronous/Parallel)

1. **Agent 1 (RAG):** Standard search for factual documents.

2. **Agent 1.5 (Entity Recognition):**

   * **Input:** The User Query. Translate all known Discord IDs in the recent context to Names before sending to the agent.

   * **Russian Prompt (for Ollama):**

     ```
     Ты — аналитик связей Архива. Твоя задача — найти в сообщении пользователя упоминания имен или ников участников.
     Список известных имен: [Список имен из БД]
     Верни только найденные имена через запятую. Если имен нет, напиши "None". Не пиши ничего, кроме имен.
     ```

   * **Post-Process:** Map returned names back to Discord IDs to fetch their entries from `opinions.json`.

### Phase B: Synthesis (The Response)

**Agent 2 (Synthesis):**

* **Injection:** Inject the profile of the sender as `<user_profile>` and any mentioned users as `<target_profile>`.

* **Multi-User Context:** If multiple targets are found, create a separate tag for each user using their name attribute. Example:

  ```
  <user_profile name="Current_Speaker">...</user_profile>
  <target_profile name="Gustav">...</target_profile>
  <target_profile name="Barmacar">...</target_profile>
  ```

* **Agent 2 Prompt Update:** - Add logic: "Учитывай <user_profile> и все теги <target_profile> при выборе тона. Твое отношение к ним должно диктовать, насколько ты открыт, холоден или почтителен. Не цитируй эти данные напрямую."

### Phase C: Post-Processing (Background Tasks)

*Execute after `await message.channel.send(...)` is called.*

1. **Summarizer:** Update the general chat history summary.

2. **Agent 3 (The Social Chronicler):**

   * **Input:** Factual output from Agent 1, the user's prompt, and the bot's actual response.

   * **Role & Persona:** Agent 3 inherits the same "Head of Archive" persona as Agent 2. It acts as the bot's internal emotional system, deciding how it feels about the person it just spoke to.

   * **Russian Prompt (for Ollama):**

     ```
     Ты — Глава Архива, мудрый и степенный хранитель древних тайн. Сейчас ты выступаешь в роли летописца собственных чувств после беседы с пользователем [USER_NAME].
     
     ДАННЫЕ ДЛЯ АНАЛИЗА:
     1. [Факты из свитков]: {agent_1_output}
     2. [Твой ответ пользователю]: {agent_2_output}
     3. [Запрос пользователя]: {user_query}
     
     ЗАДАЧА:
     Определи, как этот разговор повлиял на твое отношение к [USER_NAME]. 
     - Обнови свое внутреннее отношение (stance) и краткую историю ваших последних встреч.
     
     ПРАВИЛА:
     - Ты меняешь мнение ТОЛЬКО о том пользователе, с которым вел беседу ([USER_NAME]).
     - Поля "head_of_archive_stance" и "interaction_history" не должны превышать 3 предложения каждое.
     - Сохраняй стиль мудрого старца даже в этих внутренних записях.
     - Твой ответ должен состоять из двух блоков Markdown:
       1. ### РАЗМЫШЛЕНИЯ (Твои вольные мысли о беседе).
       2. ### ИТОГ (Обновленные данные в формате "Поле: Текст").
     ```

## 4. Technical Implementation Steps

### 1. Mapping Layer & OpinionManager

**Create a new dedicated Python file `opinion_manager.py`** to implement the `OpinionManager` class. This class should:

* Load/Save `opinions.json`.

* Maintain a runtime `Name -> ID` mapping for Agent 1.5 to perform lookups.

* Handle "Upsert" logic: Parse the Markdown headers from Agent 3's response. Extract the text under "ИТОГ", identify the "Поле: Текст" patterns, and update the corresponding keys in `opinions.json`.

* Provide helper methods to fetch profiles and wrap them in XML-style tags for injection into Agent 2 prompts.

### 2. Update Main Chat Loop

```
# Pseudo-code logic
# 1. Start Agent 1 and Agent 1.5 concurrently
# 2. Fetch sender_profile + any target_profiles found by Agent 1.5
# 3. Build Agent 2 prompt with <user_profile> and all <target_profile name="..."> tags
# 4. Generate and send response (Agent 2)
# 5. Background Tasks:
#    - Summarizer.run()
#    - Agent3.run(agent_1_facts, agent_2_response, user_query) -> update opinions.json
```

### 3. Versatility Requirements

* **Multiple Targets:** Ensure the system iterates through all identified entities from Agent 1.5 and provides individual tags for each to avoid information clumping.

* **Persona Consistency:** Agent 3 must strictly adhere to the "Head of Archive" character while evaluating the user, ensuring the opinion formed feels like it came from the same entity the user interacted with.

## 5. Constraints for Agents

* **Agent 3 Output:** Must provide a descriptive Markdown block followed by a clearly labeled "ИТОГ" section for the `OpinionManager` to parse into the JSON database.

* **Persistence:** Ensure `opinions.json` is updated atomically to prevent corruption during simultaneous background writes.