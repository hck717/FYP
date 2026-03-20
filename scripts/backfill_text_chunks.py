#!/usr/bin/env python3
"""Backfill source_file and source_name for existing text_chunks rows.

The new columns were added but existing rows have NULL values.
This script derives them from the chunk_id pattern:
  earnings_call: AAPL::earnings_call::<hash>
  broker_report: AAPL::broker_report::<hash>

and joins with textual_documents to get the real filename.
"""
import psycopg2, os

PG_HOST     = os.getenv("POSTGRES_HOST",     "postgres")
PG_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB       = os.getenv("POSTGRES_DB",       "airflow")
PG_USER     = os.getenv("POSTGRES_USER",     "airflow")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD)
cur  = conn.cursor()

# First: backfill from textual_documents join
cur.execute("""
    UPDATE text_chunks tc
    SET source_file = td.filename,
        source_name = COALESCE(
            NULLIF(tc.source_name, ''),
            CASE
                WHEN td.filename IS NOT NULL THEN REPLACE(td.filename, '.pdf', '')
                ELSE tc.chunk_id
            END
        )
    FROM textual_documents td
    WHERE tc.ticker = td.ticker
      AND td.doc_type = tc.section
      AND (
          tc.source_file IS NULL
          OR tc.source_file = ''
      )
""")
updated = cur.rowcount
conn.commit()
print(f"Updated {updated} rows from textual_documents join")

# Second: for any remaining NULL source_file, derive from chunk_id
cur.execute("""
    SELECT ticker, section, COUNT(*)
    FROM text_chunks
    WHERE (source_file IS NULL OR source_file = '')
    GROUP BY ticker, section
""")
remaining = cur.fetchall()
if remaining:
    print(f"Rows still missing source_file:")
    for r in remaining:
        print(f"  {r[0]} / {r[1]}: {r[2]} rows")
    
    cur.execute("""
        UPDATE text_chunks
        SET source_file = COALESCE(source_file, ''),
            source_name = COALESCE(source_name, chunk_id)
        WHERE source_file IS NULL OR source_file = ''
    """)
    conn.commit()
    print("Set remaining rows to use chunk_id as source_name")

# Verify
cur.execute("""
    SELECT section, COUNT(*) as total,
           SUM(CASE WHEN source_file IS NOT NULL AND source_file != '' THEN 1 ELSE 0 END) as filled
    FROM text_chunks
    WHERE section IN ('earnings_call', 'broker_report')
    GROUP BY section
""")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[2]}/{r[1]} rows with source_file populated")

conn.close()
