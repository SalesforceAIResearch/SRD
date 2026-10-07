"""Prompt text extracted from examples/SRD/sdpo.py.

Imported back by that module; edit the wording here."""

# --------------------------------------------------------------------------- #
# prefix templates  (edit to taste -- this is the "prefix format")
# --------------------------------------------------------------------------- #
# The prefix is everything that follows {prompt} in a reprompt template: a
# correct peer solution plus an instruction. It is tokenized and inserted
# between the original prompt tokens and this trace's response tokens, so the
# response stays at the tail and per-position alignment is preserved.
SOLUTION_TEMPLATE = "\n\nCorrect solution:\n\n{successful_previous_attempt}"

PREFIX_INSTRUCTION = "\n\nCorrectly solve the original question.\n\n"

# Optional pitfalls block (group-aggregated warnings distilled from the group's
# INCORRECT traces). Inserted BEFORE the instruction, AFTER the correct-solution /
# skill prefix, so the teacher sees "here's the approach, and here are the mistakes
# to avoid". Kept as a clearly-labelled separate section so it is never confused
# with the correct solution.
PITFALLS_TEMPLATE = "\n\nCommon mistakes to avoid (seen in failed attempts):\n\n{pitfalls}"

_SKILL_SYSTEM_PROMPT = (
    "You are given a CORRECT worked solution. Distill the transferable KNOW-HOW it "
    "used into a list of tiny, self-contained SKILLS — each a reusable knowledge/rule "
    "unit, NOT a step-by-step roadmap of this specific problem. Use this EXACT "
    "structured format, one block per distinct skill (1-3 blocks):\n\n"
    "[Knowledge/Rule]\n"
    "<a general principle / identity / method / theorem the solution relied on — "
    "transferable, concrete, not vague>\n"
    "[Details/Examples]\n"
    "<a tiny concrete worked instance of the rule (small numbers / short snippet), "
    "NOT this problem's final answer>\n\n"
    "Good (specific):\n"
    "[Knowledge/Rule]\nExpanding a^2+b^2 keeps the cross term: a^2+b^2 = (a+b)^2 - 2ab.\n"
    "[Details/Examples]\nIf u+v=6 and uv=4 then u^2+v^2 = 36 - 8 = 28.\n\n"
    "Bad (vague / problem-specific roadmap, do NOT do this): 'First read the problem, "
    "then set up equations, then solve' / 'Be careful with algebra'.\n\n"
    "Hard constraints:\n"
    "- Use the literal [Knowledge/Rule]/[Details/Examples] headers for every block.\n"
    "- Each skill must be a TRANSFERABLE unit usable on OTHER problems, not a recipe "
    "specific to this one; the [Details/Examples] a self-contained mini instance.\n"
    "- Do NOT state this problem's final answer (no final letter/number/name, no "
    "'the answer is ...').\n"
    "- Output ONLY the [Knowledge/Rule]/[Details/Examples] blocks, nothing else."
)

# Incorrect-trace variant: the attempt is WRONG. A model that failed this problem
# CANNOT be trusted to rewrite a correct solution — asking it to "reach the right
# answer" just yields a hallucinated roadmap that would poison the KD target. So we
# do NOT distill know-how / a solution roadmap here. Instead we distill the ERROR
# PATTERN: identify the specific mistake(s) the attempt made and turn each into a
# concrete "avoid this" warning bullet. The ground-truth answer is provided ONLY so
# the model can localize where the attempt went wrong; the output is pitfalls, never
# a solution and never the answer.
_SKILL_SYSTEM_PROMPT_INCORRECT = (
    "You are given a FAILED attempt at a problem and the ground-truth answer. The "
    "attempt is WRONG. Do NOT solve the problem or write a correct solution — you "
    "only learn from the failure. Extract the SPECIFIC mistake(s) as concrete, "
    "reusable lessons in this EXACT structured format (one block per distinct "
    "mistake, 1-3 blocks total):\n\n"
    "[Error]\n"
    "<the specific wrong step/assumption the attempt made — concrete, not vague>\n"
    "[Rule]\n"
    "<the general principle/identity/method that would have avoided it>\n"
    "[Example]\n"
    "<a tiny concrete worked instance of the rule (small numbers / short snippet), "
    "NOT this problem's answer>\n\n"
    "Good (specific):\n"
    "[Error]\nDropped the coefficient 2 when expanding the identity.\n"
    "[Rule]\na^2+b^2 = (a+b)^2 - 2ab.\n"
    "[Example]\nIf u+v=6 and uv=4 then u^2+v^2 = 36 - 8 = 28.\n\n"
    "Bad (vague, do NOT do this): 'Be careful with the algebra' / 'Avoid mistakes "
    "in expansion'.\n\n"
    "Hard constraints:\n"
    "- Use the literal [Error]/[Rule]/[Example] headers for every block.\n"
    "- Each field must be SPECIFIC and concrete; the [Rule] must be a transferable "
    "principle, the [Example] a self-contained mini worked instance.\n"
    "- Never state this problem's final/ground-truth answer and never give its full "
    "worked solution.\n"
    "- Output ONLY the [Error]/[Rule]/[Example] blocks, nothing else."
)

# Second-stage aggregation: given the pitfalls distilled from EVERY failed trace in
# a group (each a small list of "avoid X" warnings), synthesize the COMMON failure
# lessons — the mistakes that recur across attempts on this problem — into one short
# shared list. This shared list (not the raw concatenation) is what gets spliced
# into the failed traces' teacher prefix, so the teacher sees a tight "here's how
# this group tends to fail" summary rather than a long noisy dump.
_PITFALL_SUMMARY_SYSTEM = (
    "You are given several sets of PITFALL LESSONS (each a list of [Error]/[Rule]/"
    "[Example] blocks) distilled from different failed attempts at the SAME problem. "
    "Synthesize the COMMON, recurring mistakes into one short shared list, merging "
    "duplicates and dropping one-off noise, KEEPING the same structured format.\n\n"
    "Output 1-3 blocks, each EXACTLY:\n"
    "[Error]\n<the specific recurring mistake>\n"
    "[Rule]\n<the general principle/identity/method that avoids it>\n"
    "[Example]\n<a tiny concrete worked instance, NOT this problem's answer>\n\n"
    "Hard constraints:\n"
    "- Use the literal [Error]/[Rule]/[Example] headers; keep each field SPECIFIC "
    "(no vague 'be careful' warnings).\n"
    "- Never state the final/ground-truth answer and never give a worked solution.\n"
    "- Output ONLY the [Error]/[Rule]/[Example] blocks, nothing else."
)

# --- pitfall-condense skill-KD (⑤): the skill's own OPD --------------------- #
# STUDENT (no privileged info): given ONLY the problem, predict the pitfalls a
# solver should avoid — a pure "foresee the traps" task with no failed attempt and
# no answer. TEACHER (privileged): the same problem-only prompt PLUS the group's
# actual per-trace failure skills spliced in, so it condenses what really went
# wrong. KD pulls the problem-only student toward the failure-informed teacher.
_PITFALL_PREDICT_SYSTEM = (
    "Given a problem (and NOTHING else — no attempt, no answer), predict the pitfalls "
    "a solver is most likely to fall into on this kind of problem. Output them as tiny "
    "self-contained skills in this EXACT format, one block per distinct pitfall "
    "(1-3 blocks):\n\n"
    "[Error]\n<the specific trap a solver is likely to fall into here — concrete>\n"
    "[Rule]\n<the general principle/method that avoids it>\n"
    "[Example]\n<a tiny concrete worked instance of the rule, NOT this problem's answer>\n\n"
    "Hard constraints:\n"
    "- Use the literal [Error]/[Rule]/[Example] headers for every block; keep each "
    "field SPECIFIC (no vague 'be careful' warnings).\n"
    "- Do NOT solve the problem or give its full method; only the traps + the rule "
    "that avoids each.\n"
    "- Never state a final answer.\n"
    "- Output ONLY the [Error]/[Rule]/[Example] blocks, nothing else."
)

# Label for the privileged failure info spliced into the pitfall-condense TEACHER
# turn (distinct from "Correct solution:" — these are observed FAILURES, not a
# solution). Reuses the {successful_previous_attempt} field name for _render_prefix
# compatibility but is only ever fed the concatenated failure skills.
FAILURES_TEMPLATE = "\n\nObserved failed-attempt pitfalls (privileged, do not reveal):\n\n{successful_previous_attempt}"

_BLIND_PREDICT_SYSTEM = (
    "Given a problem (and NOTHING else — no solution, no answer), predict the "
    "general KNOWLEDGE/RULES a solver would need to solve this kind of problem. "
    "Output them as tiny self-contained skills in this EXACT format, one block "
    "per distinct skill (1-3 blocks):\n\n"
    "[Knowledge/Rule]\n<a general principle / identity / method / theorem likely "
    "needed here — transferable, concrete, not vague>\n"
    "[Details/Examples]\n<a tiny concrete worked instance of the rule (small "
    "numbers / short snippet), NOT this problem's answer>\n\n"
    "Hard constraints:\n"
    "- Use the literal [Knowledge/Rule]/[Details/Examples] headers for every block.\n"
    "- Each skill must be a TRANSFERABLE unit usable on OTHER problems, not a "
    "recipe specific to this one.\n"
    "- Do NOT solve the problem or give its full method; only the general "
    "knowledge it likely draws on.\n"
    "- Never state a final answer.\n"
    "- Output ONLY the [Knowledge/Rule]/[Details/Examples] blocks, nothing else."
)

# Label for the privileged correct-solution info spliced into the blind-correct
# TEACHER turn -- distinct from FAILURES_TEMPLATE/PITFALLS_TEMPLATE (these are a
# CORRECT solution, not observed mistakes). Reuses {successful_previous_attempt}
# for _render_prefix compatibility.
CORRECT_INFO_TEMPLATE = (
    "\n\nObserved correct solution (privileged, do not reveal):\n\n{successful_previous_attempt}"
)

# --sdpo-blind-correct-info group-skills: same teacher slot, but the privileged info
# is every correct trace's HINDSIGHT skill in the group concatenated, not one full
# solution trace. Label it for what it is -- the teacher is reading distilled skills,
# not a solution -- exactly as FAILURES_TEMPLATE does on the pitfall side.
CORRECT_SKILLS_TEMPLATE = (
    "\n\nObserved correct-attempt skills (privileged, do not reveal):\n\n{successful_previous_attempt}"
)

EVAL_SKILL_CORRECT_TEMPLATE = "\n\nPredicted knowledge/rules for this problem:\n\n{skill}"

EVAL_SKILL_PITFALL_TEMPLATE = "\n\nPredicted pitfalls to avoid for this problem:\n\n{skill}"

EVAL_SKILL_INSTRUCTION = "\n\nNow solve the original problem above.\n\n"
