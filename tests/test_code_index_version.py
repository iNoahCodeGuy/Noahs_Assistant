# filepath: /Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/tests/test_code_index_version.py
import time
from pathlib import Path
from assistant.core.rag_engine import RagEngine
from assistant.config.settings import Settings


def test_code_index_version_changes(tmp_path):
    s = Settings()
    s.disable_auto_rebuild = False
    engine = RagEngine(settings=s)
    initial_version = engine.code_index_version()
    assert initial_version != "none"

    # Pick a small python file to modify (create temp helper in src/utils if not exists)
    target_dir = Path('src/utils')
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / 'temp_version_probe.py'
    # Write or append a comment to change mtime
    with target_file.open('a') as f:
        f.write(f"# probe {time.time()}\n")
    time.sleep(0.2)  # ensure mtime difference on fast filesystems

    # Trigger ensure_code_index_current via retrieve_with_code
    engine.retrieve_with_code("probe change", role="Software Developer", include_code=True)
    new_version = engine.code_index_version()
    assert new_version != initial_version, f"Version did not change: {initial_version} == {new_version}"


def test_retrieve_with_code_includes_version():
    s = Settings()
    engine = RagEngine(settings=s)
    result = engine.retrieve_with_code("any query", role="Software Developer", include_code=True)
    assert 'code_index_version' in result
    assert len(result['code_index_version']) >= 8
