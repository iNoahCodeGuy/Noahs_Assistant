# One-Time Migration Scripts Archive

**Archived:** November 16, 2025 (Phase 3 Cleanup)

## Contents

### `migration_assistant.py`
- **Purpose:** Interactive helper to guide manual Supabase migration 002
- **Status:** Migration completed successfully
- **Why archived:** One-time use for initial database setup
- **Restoration:** Not needed - migration already applied to production database

### `create_test_state_helper.py`
- **Purpose:** Generated test fixture helper functions
- **Status:** Test fixtures already created and in use
- **Why archived:** Code generation script, fixtures already exist in tests/
- **Restoration:** Not needed unless regenerating test fixtures from scratch

### `generate_test_files.py`
- **Purpose:** Generated test file boilerplate
- **Status:** Test files already created
- **Why archived:** Scaffolding script, tests already implemented
- **Restoration:** Not needed unless starting new test suite from scratch

## Still Active Migration Scripts

These remain in `scripts/` for ongoing use:

- **`migrate_data_to_supabase.py`** - Reusable for career KB updates
- **`migrate_all_kb_to_supabase.py`** - Reusable for all KB updates

## Restoration Instructions

If you ever need to reference these scripts:

```bash
# View archived scripts
ls archive/old_scripts/one_time_migrations/

# Copy back to scripts/ if needed
cp archive/old_scripts/one_time_migrations/<script_name>.py scripts/
```

## Context

These scripts were created during initial project setup and database migrations. They served their purpose during development but are no longer needed for ongoing maintenance since:

1. Migration 002 is already applied to production Supabase
2. Test fixtures are already generated and committed
3. Test file structure is already established

Archived to reduce clutter while preserving historical reference.
