#!/usr/bin/env sh

set -eu

: "${SYSTEM_ONE_BASE_URL:=http://localhost:8000}"
: "${SYSTEM_ONE_API_KEY:=local-dev-key}"
: "${MODEL:=openai/gpt-4o-mini}"

curl --fail-with-body \
  --request POST "${SYSTEM_ONE_BASE_URL}/api/v1/chat/completions" \
  --header "Authorization: Bearer ${SYSTEM_ONE_API_KEY}" \
  --header "Content-Type: application/json" \
  --data "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain optimistic locking in PostgreSQL.\"}]}"
