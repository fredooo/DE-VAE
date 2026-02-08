#!/bin/bash
# lint.sh - code quality checks and formatting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

check() {
    echo "→ Checking code quality..."
    local failed=0

    ruff check . || failed=1
    black --check --diff . || failed=1
    isort --check-only --diff . || failed=1

    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All checks passed${NC}"
    else
        echo -e "${RED}Checks failed${NC}"
        exit 1
    fi
}

lint() {
    echo "→ Running linter..."
    ruff check .
}

format() {
    echo "→ Formatting code..."
    black .
    isort .
    echo "Code formatted."
}

fix() {
    echo "→ Auto-fixing issues..."

    # Don't exit on ruff errors - black/isort may fix them
    set +e
    ruff check --fix .
    set -e

    black .
    isort .

    # Final check
    echo ""
    echo "→ Running final verification..."
    if ruff check . >/dev/null 2>&1; then
        echo -e "${GREEN}All issues fixed${NC}"
    else
        echo -e "${RED}Some issues remain - run 'ruff check .' for details${NC}"
        exit 1
    fi
}

# Main
if [ $# -eq 0 ]; then
    echo "Usage: $0 {check|lint|format|fix}"
    echo ""
    echo "Commands:"
    echo "  check   - check linting and formatting (CI mode, no changes)"
    echo "  lint    - run linter only"
    echo "  format  - format code (black + isort)"
    echo "  fix     - auto-fix linting and formatting issues"
    exit 1
fi

case "$1" in
    check)
        check
        ;;
    lint)
        lint
        ;;
    format)
        format
        ;;
    fix)
        fix
        ;;
    *)
        echo "Invalid option: $1"
        echo "Usage: $0 {check|lint|format|fix}"
        exit 1
        ;;
esac
