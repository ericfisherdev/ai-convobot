#!/usr/bin/env bash
#
# Derives extraction-eval range fixtures from a companion database.
#
# The output contains real chat text, so it is written into
# `backend/tests/fixtures/local/eval/`, which is gitignored in full. Only
# this script and the harness that reads the fixtures
# (`backend/src/compaction/eval.rs`) are committed.
#
# Usage:
#   scripts/make_eval_fixtures.sh <database.db> [out-dir] [name:from-through ...]
#
# With no range arguments the four default ranges below are written. To add
# a new range, either pass it on the command line or append it to
# DEFAULT_RANGES; then run the harness once and label the items it reports
# as unlabelled into `<name>.gold.json` (see eval.rs's module docs).
set -euo pipefail

DEFAULT_RANGES=(
  "real_a:45-62"
  "real_b:63-79"
  "real_c:80-95"
  "real_full:45-95"
)

db=${1:?usage: make_eval_fixtures.sh <database.db> [out-dir] [name:from-through ...]}
out_dir=${2:-"$(dirname "$0")/../tests/fixtures/local/eval"}
shift $(( $# > 2 ? 2 : $# ))
ranges=("$@")
if [ ${#ranges[@]} -eq 0 ]; then
  ranges=("${DEFAULT_RANGES[@]}")
fi

mkdir -p "$out_dir"

user_name=$(sqlite3 "$db" "select name from user limit 1;")
companion_name=$(sqlite3 "$db" "select name from companion limit 1;")
: "${user_name:?the database has no user row}"
: "${companion_name:?the database has no companion row}"

for spec in "${ranges[@]}"; do
  name=${spec%%:*}
  bounds=${spec#*:}
  from=${bounds%%-*}
  through=${bounds##*-}

  sqlite3 -json "$db" \
    "select id, speaker_id, content from messages
     where id between $from and $through order by id;" \
  | jq --arg name "$name" \
       --arg user "$user_name" \
       --arg companion "$companion_name" \
       --arg desc "messages $from-$through from $(basename "$db")" \
       '{name: $name, description: $desc, user_name: $user,
         companion_name: $companion, messages: .}' \
  > "$out_dir/$name.json"

  count=$(jq '.messages | length' "$out_dir/$name.json")
  echo "wrote $out_dir/$name.json ($count messages)"
done
