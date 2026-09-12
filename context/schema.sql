-- ICE 差旅助手：长期记忆表结构（幂等，可重复执行）

CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    query_count BIGINT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 用户偏好：value 用 JSONB，标量/列表都能存，新增偏好类型无需改表
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    pref_type  TEXT NOT NULL,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, pref_type)
);

-- 全量聊天记录：跨会话保留
CREATE TABLE IF NOT EXISTS chat_history (
    id         BIGSERIAL PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id TEXT,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_user_time ON chat_history (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_session   ON chat_history (user_id, session_id);

-- 历史行程（trip_id 由 id 派生，不单独存列）
CREATE TABLE IF NOT EXISTS trip_history (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    origin      TEXT,
    destination TEXT,
    start_date  TEXT,
    end_date    TEXT,
    purpose     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_trip_user ON trip_history (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trip_dest ON trip_history (user_id, destination);