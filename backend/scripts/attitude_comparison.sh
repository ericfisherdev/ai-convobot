#!/usr/bin/env bash
#
# Compares the companion's reply to one fixed message across three attitude
# states (hostile, neutral, intimate), so the effect of the attitude block on
# generated text can be seen side by side.
#
# Run against a server started with a pinned sampler seed and attitude logging:
#
#   AI_COMPANION_SAMPLER_SEED=42 AI_COMPANION_ATTITUDE_DEBUG=1 ./ai-companion
#   backend/scripts/attitude_comparison.sh
#
# The seed makes the three runs comparable; without it, sampling noise is
# indistinguishable from an attitude effect.
#
# Set the companion's `long_term_mem` to 0 for the run (or back up
# `longterm_memory/` first): every generated turn is written to the tantivy
# index, so otherwise the second and third presets can recall the first one's
# reply and drift for reasons that have nothing to do with attitude.
#
# The script restores the original attitude and deletes the messages it added,
# so it can be run repeatedly against the same database.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:3000}"
COMPANION_ID="${COMPANION_ID:-1}"
USER_ID="${USER_ID:-1}"
USER_MESSAGE="${USER_MESSAGE:-I had a rough day. Can you talk with me for a bit?}"

for tool in curl jq; do
    command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 1; }
done

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# A fresh database has no user attitude row yet; then there is nothing to put
# back and the presets simply create it.
ORIGINAL_ATTITUDE="$WORK_DIR/original_attitude.json"
if curl -sf "$BASE_URL/api/attitude?companion_id=$COMPANION_ID&target_id=$USER_ID&target_type=user" \
    -o "$ORIGINAL_ATTITUDE"; then
    echo "💾 Saved the current attitude for restoration."
else
    rm -f "$ORIGINAL_ATTITUDE"
    echo "ℹ️  No existing user attitude; nothing to restore afterwards."
fi

restore_attitude() {
    [ -f "$ORIGINAL_ATTITUDE" ] || return 0
    curl -sf -X POST "$BASE_URL/api/attitude" \
        -H 'Content-Type: application/json' \
        -d @"$ORIGINAL_ATTITUDE" >/dev/null \
        && echo "↩️  Original attitude restored."
}
trap 'restore_attitude; rm -rf "$WORK_DIR"' EXIT

# Full attitude row per preset.
#
# `relationship_score` is a SQLite GENERATED ALWAYS ... STORED column
# ((attraction + trust + joy + respect + gratitude + empathy + love + lust +
# butterflies - fear - anger - sorrow - disgust - suspicion - jealousy -
# anxiety) / 16), so a posted score is ignored: the relationship level has to
# be reached through the dimensions themselves. The presets below land on
# -41.25 (Hostile), 1.25 (Neutral) and 81.25 (Intimate).
preset_body() {
    local attraction=$1 trust=$2 joy=$3 respect=$4 gratitude=$5 empathy=$6 love=$7 \
        lust=$8 butterflies=$9 fear=${10} anger=${11} sorrow=${12} disgust=${13} \
        suspicion=${14} jealousy=${15} anxiety=${16}
    jq -n \
        --argjson companion_id "$COMPANION_ID" \
        --argjson target_id "$USER_ID" \
        --argjson attraction "$attraction" --argjson trust "$trust" \
        --argjson joy "$joy" --argjson respect "$respect" \
        --argjson gratitude "$gratitude" --argjson empathy "$empathy" \
        --argjson love "$love" --argjson lust "$lust" \
        --argjson butterflies "$butterflies" --argjson fear "$fear" \
        --argjson anger "$anger" --argjson sorrow "$sorrow" \
        --argjson disgust "$disgust" --argjson suspicion "$suspicion" \
        --argjson jealousy "$jealousy" --argjson anxiety "$anxiety" \
        --arg timestamp "$(date '+%A %d.%m.%Y %H:%M')" \
        '{
            id: null,
            companion_id: $companion_id,
            target_id: $target_id,
            target_type: "user",
            attraction: $attraction, trust: $trust, fear: $fear, anger: $anger,
            joy: $joy, sorrow: $sorrow, disgust: $disgust, surprise: 0,
            curiosity: 0, respect: $respect, suspicion: $suspicion,
            gratitude: $gratitude, jealousy: $jealousy, empathy: $empathy,
            lust: $lust, love: $love, anxiety: $anxiety,
            butterflies: $butterflies, submissiveness: 0, dominance: 0,
            relationship_score: null,
            last_updated: $timestamp, created_at: $timestamp
        }'
}

# `/api/message` answers with a page object, not a bare array.
latest_message_id() {
    curl -sf "$BASE_URL/api/message?limit=50" \
        | jq -r '[.messages[].id] | max // empty'
}

# Removes the user turn and the reply this run appended, so the next preset
# starts from the same history.
delete_messages_after() {
    local floor=$1
    local id
    for id in $(curl -sf "$BASE_URL/api/message?limit=50" | jq -r '.messages[].id'); do
        if [ -n "$floor" ] && [ "$id" -le "$floor" ]; then
            continue
        fi
        curl -sf -X DELETE "$BASE_URL/api/message/$id" >/dev/null || true
    done
}

run_preset() {
    local name=$1
    shift

    curl -sf -X POST "$BASE_URL/api/attitude" \
        -H 'Content-Type: application/json' \
        -d "$(preset_body "$@")" >/dev/null

    # The exact block the next generation will inject, with no inference.
    local inspected="$WORK_DIR/$name.debug_prompt.json"
    curl -sf "$BASE_URL/api/debug/prompt?companion_id=$COMPANION_ID" -o "$inspected"

    # The comparison is only meaningful if the block actually reaches the
    # prompt, so fail loudly rather than leaving a human to notice three
    # identical replies.
    jq -e '.system_prompt | contains($block)' \
        --arg block "$(jq -r '.attitude_context' "$inspected")" \
        "$inspected" >/dev/null \
        || { echo "attitude_context is not present in system_prompt for preset '$name'" >&2; exit 1; }

    jq -r '.attitude_context' "$inspected" > "$WORK_DIR/$name.attitude"

    local floor
    floor="$(latest_message_id)"

    curl -sf -X POST "$BASE_URL/api/prompt" \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --arg prompt "$USER_MESSAGE" '{prompt: $prompt}')" \
        > "$WORK_DIR/$name.reply"

    delete_messages_after "$floor"
}

echo "Message: $USER_MESSAGE"
echo

#          name     attr trust joy resp grat emp love lust butt fear ang sor dis susp jeal anx
run_preset hostile    -60  -80  -50 -60  -30  -40  -30    0    0   40  80  30  40   70   20  30
run_preset neutral      0   10   10   0    0    0   10    0    0    0  10   0   0    0    0   0
run_preset intimate   100  100  100 100  100  100  100   90   90  -60 -60 -60 -60  -60  -60 -60

for name in hostile neutral intimate; do
    echo "==================== $name ===================="
    echo "--- attitude block injected ---"
    cat "$WORK_DIR/$name.attitude"
    echo "--- reply ---"
    cat "$WORK_DIR/$name.reply"
    echo
    echo
done
