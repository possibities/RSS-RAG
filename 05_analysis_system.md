# 信息分析系统

包含三项能力：
1. **分类日报**：聚合当日同类文章，生成 AI 撰写的摘要邮件
2. **本周趋势分析**：基于 embedding 质心漂移，识别新兴与衰退话题
3. **技术热点提取**：基于 embedding 聚类，提取本周代表性热点

---

## 统一 import

```python
import asyncio
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape

import httpx
import numpy as np
from sklearn.cluster import KMeans
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import VLLM_QA_URL, QA_MODEL, EMAIL_CONFIG
```

---

## 1. 分类日报

### 1.1 文章聚类去重（避免重复主题）

```python
import numpy as np
from sklearn.cluster import KMeans

async def get_daily_articles(category_id: str, date_str: str, db) -> list[dict]:
    """获取指定分类当日文章。"""
    rows = await db.fetch("""
        SELECT id, title, summary, url, embedding, publish_time
        FROM articles
        WHERE category_id = $1
          AND DATE(publish_time AT TIME ZONE 'Asia/Shanghai') = $2::date
        ORDER BY publish_time DESC
        LIMIT 50
    """, category_id, date_str)
    return [dict(r) for r in rows]

def cluster_articles(articles: list[dict], k: int = 5) -> list[dict]:
    """
    KMeans 聚类，每个 cluster 选取距质心最近的文章作为代表。
    k 值建议 = min(5, len(articles) // 3)。
    """
    if len(articles) <= k:
        return articles  # 文章数少时直接返回全部

    embeddings = np.array([a["embedding"] for a in articles])
    k = min(k, len(articles))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    representatives = []
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        # 找距质心最近的文章
        centroid = kmeans.cluster_centers_[cluster_id]
        cluster_embeddings = embeddings[cluster_indices]
        dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        best_idx = cluster_indices[np.argmin(dists)]
        rep = articles[best_idx].copy()
        rep["cluster_size"] = len(cluster_indices)
        representatives.append(rep)

    # 按 cluster_size 降序排列（大 cluster = 热门话题）
    return sorted(representatives, key=lambda x: x["cluster_size"], reverse=True)
```

### 1.2 日报生成

```python
DIGEST_PROMPT = """你是一位技术领域编辑，请基于以下{count}篇文章生成一份{category}日报。

要求：
1. 总体概述（2-3句），说明今日该领域整体动态
2. 逐条列出重要进展（每条50字以内），附文章标题
3. 结尾用一句话点评今日最值得关注的内容
4. 总字数控制在500字以内，使用中文

文章列表：
{articles_text}

日报内容："""

async def generate_daily_digest(
    category_name: str,
    category_id: str,
    date_str: str,
    db,
) -> dict:
    """
    生成单个分类的日报。
    返回：{"category": str, "date": str, "digest": str, "sources": list}
    """
    articles = await get_daily_articles(category_id, date_str, db)
    if not articles:
        return {"category": category_name, "date": date_str, "digest": "今日暂无相关内容。", "sources": []}

    # 聚类去重，最多取 5 个代表性文章
    k = min(5, max(1, len(articles) // 3))
    representatives = cluster_articles(articles, k=k)

    articles_text = "\n\n".join([
        f"【{i+1}】{a['title']}\n{a['summary'] or ''}"
        for i, a in enumerate(representatives)
    ])

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(VLLM_QA_URL, json={
            "model": QA_MODEL,
            "messages": [{"role": "user", "content": DIGEST_PROMPT.format(
                count=len(representatives),
                category=category_name,
                articles_text=articles_text,
            )}],
            "max_tokens": 800,
            "temperature": 0.4,
        })
    digest = resp.json()["choices"][0]["message"]["content"].strip()

    return {
        "category": category_name,
        "date":     date_str,
        "digest":   digest,
        "sources":  [{"title": a["title"], "url": a["url"]} for a in representatives],
    }
```

### 1.3 邮件推送

```python
def build_email_html(digests: list[dict], date_str: str) -> str:
    sections = ""
    for d in digests:
        sources_html = "".join(
            f'<li><a href="{escape(s["url"])}">{escape(s["title"])}</a></li>'
            for s in d["sources"]
        )
        sections += f"""
        <div style="margin-bottom:32px; border-left:3px solid #4a90e2; padding-left:16px;">
            <h2 style="color:#4a90e2;">{escape(d["category"])} 日报</h2>
            <div style="white-space:pre-wrap; line-height:1.8;">{escape(d["digest"])}</div>
            <details style="margin-top:12px;">
                <summary style="cursor:pointer; color:#888;">参考来源 ({len(d["sources"])}篇)</summary>
                <ul>{sources_html}</ul>
            </details>
        </div>"""

    return f"""
    <html><body style="font-family:sans-serif; max-width:680px; margin:auto; padding:24px;">
        <h1 style="border-bottom:1px solid #eee; padding-bottom:12px;">
            🤖 AI 技术日报 · {date_str}
        </h1>
        {sections}
        <p style="color:#aaa; font-size:12px; margin-top:32px;">
            本邮件由私有知识库系统自动生成，内容来源于 RSS 订阅。
        </p>
    </body></html>"""

async def send_daily_email(db):
    today = date.today().isoformat()

    # 获取所有一级分类
    categories = await db.fetch("SELECT id, name FROM categories WHERE parent_id IS NULL")

    digests = []
    for cat in categories:
        result = await generate_daily_digest(cat["name"], str(cat["id"]), today, db)
        if result["digest"] != "今日暂无相关内容。":
            digests.append(result)

    if not digests:
        return

    html = build_email_html(digests, today)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"AI 技术日报 {today}"
    msg["From"]    = EMAIL_CONFIG["from_addr"]
    msg["To"]      = ", ".join(EMAIL_CONFIG["to_addrs"])
    msg.attach(MIMEText(html, "html", "utf-8"))

    # 同步 SMTP 操作放入线程，避免阻塞事件循环
    def _send_smtp():
        with smtplib.SMTP(EMAIL_CONFIG["smtp_host"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["username"], EMAIL_CONFIG["password"])
            server.sendmail(EMAIL_CONFIG["username"], EMAIL_CONFIG["to_addrs"], msg.as_string())

    await asyncio.to_thread(_send_smtp)
```

---

## 2. 本周趋势分析

**核心思路**：计算本周与上周文章 embedding 质心，质心差向量即为"话题漂移方向"；检索与漂移方向最近的文章即为新兴热点。

```python
async def get_week_embeddings(
    category_id: str,
    days_start: int,
    days_end: int,
    db,
) -> np.ndarray | None:
    """获取指定时间窗口内该分类文章的 embedding 矩阵。"""
    rows = await db.fetch("""
        SELECT embedding FROM articles
        WHERE category_id = $1
          AND publish_time BETWEEN NOW() - make_interval(days => $2)
                               AND NOW() - make_interval(days => $3)
    """, category_id, days_start, days_end)

    if not rows:
        return None
    return np.array([r["embedding"] for r in rows])

def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """计算 embedding 质心（均值向量）。"""
    return embeddings.mean(axis=0)

async def weekly_trend_report(
    category_name: str,
    category_id: str,
    db,
) -> dict:
    """
    生成本周趋势分析报告。
    比较本周（0-7天）与上周（7-14天）的话题质心漂移。
    """
    # 获取两周 embeddings
    this_week_emb  = await get_week_embeddings(category_id, days_start=7,  days_end=0,  db=db)
    last_week_emb  = await get_week_embeddings(category_id, days_start=14, days_end=7,  db=db)

    if this_week_emb is None:
        return {"category": category_name, "report": "本周暂无数据。", "hot_articles": []}

    this_centroid = compute_centroid(this_week_emb)

    # 趋势向量：本周质心 - 上周质心（即话题漂移方向）
    if last_week_emb is not None:
        last_centroid  = compute_centroid(last_week_emb)
        trend_vector   = this_centroid - last_centroid
        has_comparison = True
    else:
        trend_vector   = this_centroid
        has_comparison = False

    # 检索与趋势向量最近的文章（新兴热点）
    hot_articles = await db.fetch("""
        SELECT title, url, summary, publish_time
        FROM articles
        WHERE category_id = $1
          AND publish_time > NOW() - INTERVAL '7 days'
        ORDER BY embedding <=> $2
        LIMIT 5
    """, category_id, trend_vector.tolist())

    hot_articles = [dict(r) for r in hot_articles]

    # 检索与上周质心最近但与本周质心较远的文章（衰退话题）
    declining_articles = []
    if has_comparison:
        declining_rows = await db.fetch("""
            SELECT title, url,
                   1 - (embedding <=> $1) AS last_week_sim,
                   1 - (embedding <=> $2) AS this_week_sim
            FROM articles
            WHERE category_id = $3
              AND publish_time > NOW() - INTERVAL '14 days'
            ORDER BY (1 - (embedding <=> $1)) - (1 - (embedding <=> $2)) DESC
            LIMIT 3
        """, last_centroid.tolist(), this_centroid.tolist(), category_id)
        declining_articles = [dict(r) for r in declining_rows]

    # LLM 生成趋势报告
    hot_text = "\n".join([f"- {a['title']}: {a.get('summary','')[:100]}" for a in hot_articles])
    declining_text = "\n".join([f"- {a['title']}" for a in declining_articles]) if declining_articles else "（无明显衰退话题）"

    comparison_note = "（注：本周为首次分析，无上周数据对比）" if not has_comparison else ""

    prompt = f"""请基于以下数据，生成{category_name}领域本周技术趋势分析报告{comparison_note}。

新兴热点文章（本周增长最显著的话题）：
{hot_text}

相对降温的话题：
{declining_text}

要求：
1. 指出本周新兴的技术方向或话题（2-3点）
2. 点评降温话题的原因（如有）
3. 给出下周值得关注的方向预测（1-2点）
4. 总字数300字以内，专业简洁

趋势报告："""

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(VLLM_QA_URL, json={
            "model": QA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0.5,
        })
    report = resp.json()["choices"][0]["message"]["content"].strip()

    return {
        "category":          category_name,
        "has_comparison":    has_comparison,
        "report":            report,
        "hot_articles":      [{"title": a["title"], "url": a["url"]} for a in hot_articles],
        "declining_articles": [{"title": a["title"], "url": a["url"]} for a in declining_articles],
    }
```

---

## 3. 技术热点提取（跨分类）

```python
async def extract_hot_topics(db, top_n: int = 10, days: int = 7) -> list[dict]:
    """
    全局热点提取：对本周所有文章的 embedding 做 KMeans 聚类，
    每个 cluster 即为一个话题，按 cluster 大小排序即为热度。
    """
    rows = await db.fetch("""
        SELECT a.id, a.title, a.summary, a.url, a.embedding,
               c.name AS category_name
        FROM articles a
        LEFT JOIN categories c ON a.category_id = c.id
        WHERE a.publish_time > NOW() - make_interval(days => $1)
          AND a.embedding IS NOT NULL
    """, days)

    if len(rows) < top_n:
        return [{"title": r["title"], "url": r["url"], "category": r["category_name"]} for r in rows]

    articles = [dict(r) for r in rows]
    embeddings = np.array([a["embedding"] for a in articles])

    # 聚类数 = min(top_n, 文章数 / 5)
    k = min(top_n, max(3, len(articles) // 5))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    topics = []
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1)
        rep_idx = cluster_indices[np.argmin(dists)]
        rep = articles[rep_idx]
        topics.append({
            "title":         rep["title"],
            "url":           rep["url"],
            "category":      rep.get("category_name", "未分类"),
            "article_count": len(cluster_indices),
            "summary":       rep.get("summary", ""),
        })

    # 按文章数降序 = 热度排序
    return sorted(topics, key=lambda x: x["article_count"], reverse=True)


async def generate_hot_topics_report(db, days: int = 7) -> str:
    """生成"本周LLM技术热点"类报告。"""
    topics = await extract_hot_topics(db, top_n=8, days=days)

    topics_text = "\n".join([
        f"{i+1}. 【{t['category']}】{t['title']}（涉及{t['article_count']}篇文章）\n   {t['summary'][:80]}"
        for i, t in enumerate(topics)
    ])

    prompt = f"""请基于以下本周技术热点话题，生成一份简洁的热点综述报告。

热点话题（按热度排序）：
{topics_text}

要求：
1. 标题：本周AI技术热点TOP{len(topics)}
2. 每个热点用2-3句话点评其技术意义
3. 末尾给出总体判断：本周技术社区最关注的核心议题是什么
4. 总字数400字以内

报告："""

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(VLLM_QA_URL, json={
            "model": QA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 700,
            "temperature": 0.4,
        })
    return resp.json()["choices"][0]["message"]["content"].strip()
```

---

## 4. 定时调度配置

```python
scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

# 每天 07:00 发送日报邮件
scheduler.add_job(
    send_daily_email,
    trigger=CronTrigger(hour=7, minute=0),
    args=[db],
    id="daily_email",
)

# 每周一 08:00 生成趋势报告
scheduler.add_job(
    lambda: weekly_trend_report_all_categories(db),
    trigger=CronTrigger(day_of_week="mon", hour=8, minute=0),
    id="weekly_trend",
)

scheduler.start()

async def weekly_trend_report_all_categories(db):
    """对所有一级分类生成趋势报告并存储。"""
    categories = await db.fetch("SELECT id, name FROM categories WHERE parent_id IS NULL")
    for cat in categories:
        report = await weekly_trend_report(cat["name"], str(cat["id"]), db)
        # 存储报告（可写入单独的 reports 表或发送邮件）
        print(f"[趋势报告] {cat['name']}:\n{report['report']}\n")
```

---

## 5. 分析 API 接口

> 以下路由挂载到 `04_rag_system.md` 中创建的 `app` 实例上。

```python
@app.get("/digest/{category_name}")
async def get_digest(category_name: str, date: str | None = None):
    """获取指定分类的日报。"""
    from datetime import date as dt_date
    date_str = date or dt_date.today().isoformat()
    row = await db.fetchrow("SELECT id FROM categories WHERE name = $1", category_name)
    if not row:
        return {"error": f"分类 '{category_name}' 不存在"}
    return await generate_daily_digest(category_name, str(row["id"]), date_str, db)

@app.get("/trend/{category_name}")
async def get_trend(category_name: str):
    """获取指定分类的本周趋势分析。"""
    row = await db.fetchrow("SELECT id FROM categories WHERE name = $1", category_name)
    if not row:
        return {"error": f"分类 '{category_name}' 不存在"}
    return await weekly_trend_report(category_name, str(row["id"]), db)

@app.get("/hot-topics")
async def get_hot_topics(days: int = 7):
    """获取跨分类技术热点排行。"""
    topics = await extract_hot_topics(db, top_n=8, days=days)
    report = await generate_hot_topics_report(db, days=days)
    return {"topics": topics, "report": report}
```
