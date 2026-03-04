# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Early-stage Python project. No build system, dependencies, or source code added yet.

Expected layout (based on `.gitignore`):
- `.venv/` — Python virtual environment
- `.env` — local environment variables (not tracked)

## Language Preferences

- **Chinese** for explanations, summaries, document reading, and conceptual answers.
- **English** for all code, comments, docstrings, commit messages, file names, and shell commands.
- When both are needed: Chinese explanation first, then English code/commands in separate fenced blocks.
- All identifiers (variables, functions, classes, directories) must be in English.

## Workflow

- Prefer small, reviewable diffs. Do not change unrelated files.
- Do not delete or rename files unless explicitly requested.

## Defaults

- This repository is **public**: never introduce secrets, credentials, tokens, or personal paths into tracked files.
