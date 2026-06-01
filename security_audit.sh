#!/bin/bash
# Security audit script for spacial-boxcounting
# Run before releases and PRs to catch issues early.
set -e

echo "🔍 Running security audit..."

# SAST — static analysis for common Python security issues
if command -v bandit &>/dev/null; then
    echo "  → bandit (SAST)..."
    bandit -r spacial_boxcounting/ --exclude ./tests,./.venv,./deprecated
else
    echo "  ⚠ bandit not installed — skipping SAST"
fi

# Dependency vulnerability scan
if command -v safety &>/dev/null; then
    echo "  → safety (dependency scan)..."
    safety check -r requirements.lock
else
    echo "  ⚠ safety not installed — install with: pip install safety"
fi

# Git history secret scan
if command -v gitleaks &>/dev/null; then
    echo "  → gitleaks (secret scan)..."
    gitleaks detect --source . --verbose
else
    echo "  ⚠ gitleaks not installed — see https://github.com/gitleaks/gitleaks"
fi

echo "✅ Security audit complete"
