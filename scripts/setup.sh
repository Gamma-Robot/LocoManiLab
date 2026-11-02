#!/usr/bin/env bash
# Local core setup and explicit, optional source checkout plans.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LOCOMANILAB_SETUP_PYTHON:-${SOMA_SETUP_PYTHON:-python3}}"
TARGET="${1:-}"
if [[ -z "$TARGET" || "$TARGET" == "--help" || "$TARGET" == "-h" ]]; then
  cat <<'HELP'
Usage:
  bash scripts/setup.sh lab [--plan]     Create .venv and install the data-enabled CLI.
  bash scripts/setup.sh core [--plan]    Alias for lab; no model environments required.
  bash scripts/setup.sh <profile>        Show optional source checkout plan; no downloads.
  bash scripts/setup.sh <profile> --apply
                                       Initialize missing selected submodules, code only.
  python3 scripts/setup/profiles.py --list

Model environments, weights, SDKs, and simulation assets are provisioned separately.
Run this file from a source checkout. The public CPU quickstart is in README.md.
HELP
  exit 0
fi
shift

case "$TARGET" in
  lab|locomanilab|soma|core|data)
    if [[ "${1:-}" == "--plan" ]]; then
      echo "Create $ROOT/.venv with $PYTHON; install editable $ROOT[data]."
      echo "No submodules, model environments, weights, SDKs, or scene assets are installed."
      exit 0
    fi
    if [[ $# -gt 0 ]]; then
      echo "Unknown core setup option: $1 (use --plan or no option)." >&2
      exit 2
    fi
    if [[ ! -f "$ROOT/.venv/pyvenv.cfg" ]]; then
      "$PYTHON" -m venv "$ROOT/.venv"
    fi
    "$ROOT/.venv/bin/python" -m pip install -e "$ROOT[data]"
    echo
    echo "Core installation finished. This script does not change your parent shell."
    printf 'Activate it with: source %q\n' "$ROOT/.venv/bin/activate"
    printf 'Or run directly: %q --help\n' "$ROOT/.venv/bin/lab"
    ;;
  humanoid-benchmark|tele-benchmark) "$PYTHON" "$ROOT/scripts/setup/profiles.py" sim "$@" ;;
  g1_backend|sonic-collect) "$PYTHON" "$ROOT/scripts/setup/profiles.py" real "$@" ;;
  sonic-deploy) "$PYTHON" "$ROOT/scripts/setup/profiles.py" gr00t-real "$@" ;;
  *) "$PYTHON" "$ROOT/scripts/setup/profiles.py" "$TARGET" "$@" ;;
esac
