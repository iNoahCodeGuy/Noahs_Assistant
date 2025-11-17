# Legacy Scripts Archive

This directory contains one-time migration scripts and deprecated utilities.

## Archived November 16, 2025

### Database Migrations (Completed)
- `run_migration*.py` - Historical Supabase schema migrations
- `run_session_id_migration.py` - Session ID migration (complete)
- `migrate_to_typeddict.py` - Code refactor to TypedDict (complete)

### Knowledge Base Population (One-Time)
- `add_architecture_kb.py` - Initial architecture KB load
- `add_technical_kb.py` - Initial technical KB load
- `add_impressive_questions.py` - Sample questions
- `add_product_questions.py` - Product questions

### Deprecated System Rewrites
- `setup_modular_system.py` - Old modular system setup
- `replace_analytics.py` - Analytics system rewrite (complete)
- `phase3_setup_wizard.py` - Phase 3 planning tool
- `setup_phase3.py` - Phase 3 setup (now in production)

## Active Scripts

See `scripts/` directory for current utilities:
- `migrate_data_to_supabase.py` - Reusable KB migration
- `test_*.py` - Active testing suite
- `verify_*.py` - CI/deployment verification
