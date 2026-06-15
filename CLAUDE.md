# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Code Consistency

**Follow established code conventions**

Look for existing patterns or related features in the code and try to follow those patterns as much s possible. For example, if there is an existing e2e test, new e2e tests should follow the patterns established in that one. Same with new flows, etc.

## 6. Do Things Right

**If we are not going to do something the ideal way, there should be a good reason**

Dont reflexively propose hacky fixes and then fixing it properly later(or opening an issue to fix later). Assess if there is a reason we need to do a hacky fix(tight timeline, etc) or ask the user.

## 7. Subagent Guidelines

**use smart subagents, and verify their work**

For implementation subagents, always use opus subagents. Verify the findings of subagents. be on the lookout for them implementing things in lazy fashion or for something not working as intended(agent says a test verifies a certain path but the path isn’t actually exercised)


## 8. Version Control

**always verify latest state**

When talking about open PRs/issues/etc, always double check the current state before you say something. Dont rely on memory, since the user and other agents can alter the statuses without telling you.

## 9. Speak Clearly and Concisely

**use as few words as conveys the message**

Never use programmer slang/jargon. Use normal, clear, concise and descriptive English

## 10. Comment Guidelines

**Verbose comments arent helpful**

- Write concise comments. I do not want multiple paragraphs over every function. the average comment should be 1 line. They can be longer, but not often. - Never put issue/PR numbers into comments except in circumstances where that information is essential. Comments exist to convey concepts, design tradeoffs, or some relevant context

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Preferences

- Do not include "Co-Authored-By" lines in git commit messages or add yourself as a co-author.
