-- SignalRank — pgvector + FTS schema
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS jobs CASCADE;

CREATE TABLE jobs (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    company         TEXT NOT NULL,
    location        TEXT,
    description     TEXT NOT NULL,
    skills          JSONB DEFAULT '[]'::jsonb,
    source          TEXT DEFAULT 'synthetic',
    tsv             TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(description,''))) STORED,
    embedding       VECTOR(384),
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX jobs_tsv_idx ON jobs USING GIN (tsv);
-- HNSW cosine index — tuned for 500 rows, scales to 50k without reindex
CREATE INDEX jobs_vec_idx ON jobs USING HNSW (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Optional: qrels for eval (also kept as jsonl on disk)
DROP TABLE IF EXISTS qrels CASCADE;
CREATE TABLE qrels (
    cv_id   TEXT NOT NULL,
    job_id  INT REFERENCES jobs(id) ON DELETE CASCADE,
    relevance INT NOT NULL CHECK (relevance IN (0,1,2)),
    PRIMARY KEY (cv_id, job_id)
);
