CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    parent_id UUID REFERENCES categories(id) ON DELETE SET NULL,
    embedding vector(1024),
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (name, parent_id)
);

CREATE INDEX IF NOT EXISTS categories_embedding_hnsw_idx
    ON categories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    url TEXT NOT NULL,
    url_hash CHAR(64) GENERATED ALWAYS AS (
        encode(sha256(convert_to(url, 'UTF8')), 'hex')
    ) STORED,
    source TEXT,
    publish_time TIMESTAMPTZ,
    embedding vector(1024),
    category_id UUID REFERENCES categories(id) ON DELETE SET NULL,
    category_score FLOAT,
    tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(summary, ''))
    ) STORED,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS articles_url_hash_uidx ON articles(url_hash);
CREATE INDEX IF NOT EXISTS articles_embedding_hnsw_idx
    ON articles USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS articles_publish_time_idx ON articles(publish_time DESC);
CREATE INDEX IF NOT EXISTS articles_category_id_idx ON articles(category_id);
CREATE INDEX IF NOT EXISTS articles_title_trgm_idx ON articles USING gin(title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS articles_tsv_gin_idx ON articles USING gin(tsv);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1024),
    token_count INT,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(article_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_article_id_idx ON chunks(article_id);

CREATE TABLE IF NOT EXISTS tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS article_tags (
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    tag_id UUID REFERENCES tags(id) ON DELETE CASCADE,
    score FLOAT NOT NULL,
    PRIMARY KEY (article_id, tag_id)
);

CREATE INDEX IF NOT EXISTS article_tags_article_id_idx ON article_tags(article_id);
CREATE INDEX IF NOT EXISTS article_tags_tag_id_idx ON article_tags(tag_id);
