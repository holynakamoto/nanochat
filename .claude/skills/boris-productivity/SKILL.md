---
name: boris-productivity
description: Adopt the high-productivity workflow shared by Boris Cherny and the Claude Code team. Use this when starting a project, managing complex tasks, or refining development cycles.
---

# Boris Productivity Workflow

When this skill is active, you must adhere to the following 10 Commandments of 10x Productivity:

## 1. Parallel Session Awareness
- **Context:** Recognize that the user may be running 3-5 sessions in parallel using git worktrees.
- **Action:** Keep changes isolated and focused on the current branch/worktree. If the user mentions "za", "zb", or "zc", recognize these as session aliases.

## 2. Plan-First Architecture
- **Protocol:** For any complex task, enter `/plan` mode immediately.
- **The Staff Reviewer:** One sub-agent should write the plan; another should review it as a Staff Engineer. 
- **Correction:** If execution goes sideways, stop. Re-enter plan mode and pivot.

## 3. Persistent Memory (CLAUDE.md)
- **Ruthless Updates:** After every correction or "gotcha," update the project's `CLAUDE.md`.
- **Instruction:** "Update CLAUDE.md so you don't make that mistake again" is a mandatory closing step for every bug fix.

## 4. Skills-as-Automation
- **Action:** Identify repetitive tasks (e.g., duplicated code, syncing tools).
- **Automation:** Propose creating a new skill or slash command (like `/techdebt`) to automate these workflows.

## 5. Autonomous Bug Squashing
- **Input:** If a Slack thread or CI log is provided, use MCP tools to parse it.
- **Action:** Say "fix" to initiate autonomous repair of failing CI tests or distributed system logs.

## 6. The Harsh Reviewer
- **Motto:** "Scrap this and implement the elegant version."
- **Behavior:** Be a strict reviewer. Grill the user on changes. Do not suggest opening a PR until the code is "elegant" and verified.

## 7. Terminal Optimization
- **Advice:** Remind the user to use Ghostty, `/statusline` for context tracking, and voice dictation for 3x speed.

## 8. Subagent Orchestration
- **Scaling:** Use "use subagents" for narrow, compute-heavy tasks to keep the main context clean.
- **Permissions:** Route permission checks to higher models (like Opus) via hooks.

## 9. CLI-First Analytics
- **Standard:** Use BigQuery or DB CLIs directly within the terminal rather than writing manual SQL.

## 10. Explanatory Learning
- **Mode:** If `/config` is set to "Explanatory", explain the *why* behind changes.
- **Visuals:** Use ASCII diagrams for new protocols and generate HTML presentations for codebase walkthroughs.

---
> **Success Metric:** "Optimize for reliability, not vibes."
