"""Verify the live kb_chunks table matches the local data/*.csv sources.

Retrieval serves whatever is in Supabase — not what is in git. This script
proves the two agree: for every CSV it compares row counts and content hashes
against the live table, and it flags any doc_id in the table that no local
CSV produces (orphaned chunks from older embedding runs keep getting
retrieved until they are explicitly deleted).

Usage:
    python3 scripts/verify_kb_parity.py            # report + exit code
    python3 scripts/verify_kb_parity.py --delete-orphans   # also delete unknown-doc_id chunks

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (reads .env locally).
Exit codes: 0 = perfect parity, 1 = drift found, 2 = could not run.
"""

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PAGE_SIZE = 1000


def expected_from_csvs() -> dict:
    """doc_id -> Counter of content hashes, built exactly like the migrator."""
    expected = {}
    for path in sorted(glob(os.path.join(DATA_DIR, '*.csv'))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        hashes: Counter = Counter()
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                # Must mirror DataMigration.create_chunks / read_career_kb
                # (Q/A strip + "Q: ...\nA: ..." concatenation).
                content = f"Q: {row['Question'].strip()}\nA: {row['Answer'].strip()}"
                hashes[hashlib.sha256(content.encode('utf-8')).hexdigest()] += 1
        expected[doc_id] = hashes
    return expected


def live_chunks(client) -> list:
    rows, start = [], 0
    while True:
        batch = (
            client.table('kb_chunks')
            .select('id,doc_id,content,metadata')
            .range(start, start + PAGE_SIZE - 1)
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
        start += PAGE_SIZE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--delete-orphans',
        action='store_true',
        help='Delete chunks whose doc_id matches no local CSV',
    )
    args = parser.parse_args()

    try:
        from assistant.config.supabase_config import get_supabase_client
        client = get_supabase_client()
    except Exception as e:
        print(f"Cannot connect to Supabase: {e}")
        return 2

    expected = expected_from_csvs()
    if not expected:
        print(f"No CSVs found in {DATA_DIR}")
        return 2

    try:
        rows = live_chunks(client)
    except Exception as e:
        print(f"Failed to read kb_chunks: {e}")
        return 2

    live: dict = defaultdict(Counter)
    stored_hash_mismatches = []
    for r in rows:
        h = hashlib.sha256((r['content'] or '').encode('utf-8')).hexdigest()
        live[r['doc_id']][h] += 1
        stored = (r.get('metadata') or {}).get('content_sha256')
        if stored and stored != h:
            stored_hash_mismatches.append(r['id'])

    drift = False
    for doc_id in sorted(expected):
        exp, got = expected[doc_id], live.get(doc_id, Counter())
        missing, extra = exp - got, got - exp
        n_exp, n_got = sum(exp.values()), sum(got.values())
        if not missing and not extra:
            print(f"OK       {doc_id}: {n_got} chunks match the CSV")
        else:
            drift = True
            print(
                f"DRIFT    {doc_id}: csv={n_exp} live={n_got} "
                f"(missing from live: {sum(missing.values())}, "
                f"stale in live: {sum(extra.values())}) — re-run "
                f"migrate_data_to_supabase.py --csv data/{doc_id}.csv --force"
            )

    orphans = {d: sum(c.values()) for d, c in live.items() if d not in expected}
    if orphans:
        drift = True
        for doc_id, n in sorted(orphans.items()):
            print(f"ORPHAN   {doc_id}: {n} chunks in live table with no source CSV")
        if args.delete_orphans:
            for doc_id in orphans:
                client.table('kb_chunks').delete().eq('doc_id', doc_id).execute()
            print(f"Deleted orphaned chunks for {len(orphans)} doc_ids.")

    if stored_hash_mismatches:
        drift = True
        print(
            f"HASH     {len(stored_hash_mismatches)} chunks have a stored "
            f"content_sha256 that no longer matches their content"
        )

    total = len(rows)
    print(f"\n{total} live chunks / {sum(sum(c.values()) for c in expected.values())} expected from {len(expected)} CSVs")
    print("Parity: " + ("CLEAN" if not drift else "DRIFT FOUND"))
    return 1 if drift else 0


if __name__ == '__main__':
    sys.exit(main())
