# Per-response cap, applied to every loop turn and to commit(). Adaptive
# thinking at THINKING_EFFORT plus a tool call all has to fit inside it; a
# response that runs over comes back with stop_reason == "max_tokens" rather
# than raising, so truncation is silent.
MAX_TOKENS = 10_000

# Must stay identical to src/basic_prompting/constants.py. The benchmark's
# claim is the same model with and without tools, so drift between the two
# turns the comparison into a model comparison.
MODEL = "claude-opus-5"

# The Files API beta is required on the upload *and* on every messages call
# that references the resulting file_id — hence one list passed to all three.
BETAS = ["files-api-2025-04-14"]

# The per-frame cost ceiling: ~11 turns is typical, so this is headroom rather
# than a target. Exhausting it is not a failure — run()'s for/else falls
# through to commit(), which is what keeps a slow frame from returning nothing.
MAX_TURNS = 15

# output_config effort, not a thinking budget (budget_tokens is rejected on
# this model). It deepens reasoning inside the same MAX_TOKENS, so the two
# constants are coupled: raising this without raising that invites truncation.
THINKING_EFFORT = "high"
