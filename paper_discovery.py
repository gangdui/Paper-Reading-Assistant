import json
import math
import os
import re
from datetime import date
from difflib import SequenceMatcher
from typing import Any
import xml.etree.ElementTree as ET

from openai import OpenAI
import requests


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_SEARCH_URL = "https://export.arxiv.org/api/query"
DBLP_PUBLICATION_SEARCH_URL = "https://dblp.org/search/publ/api"
DEFAULT_API_TIMEOUT_SECONDS = 8
SEMANTIC_SCHOLAR_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "authors",
        "year",
        "venue",
        "abstract",
        "citationCount",
        "url",
        "openAccessPdf",
    ]
)
SOURCE_ARXIV = "arxiv"
SOURCE_SEMANTIC = "semantic_scholar"
SOURCE_DBLP = "dblp"
DEFAULT_SELECTED_SOURCES = [SOURCE_ARXIV, SOURCE_SEMANTIC, SOURCE_DBLP]

QUALITY_VENUES = {
    "acl",
    "emnlp",
    "naacl",
    "neurips",
    "icml",
    "iclr",
    "cvpr",
    "iccv",
    "eccv",
    "aaai",
    "ijcai",
    "sigir",
    "kdd",
    "www",
    "usenix security",
    "ieee s&p",
    "ccs",
    "ndss",
}

DEFAULT_WATERMARKING_QUERIES = [
    "AI-generated content watermarking",
    "generative AI watermarking",
    "diffusion model watermarking",
    "LLM watermarking",
    "model watermarking",
    "neural network watermarking",
]

MOCK_PAPERS = [
    {
        "paperId": "mock:aigc-watermarking-1",
        "title": "Mock Example: Watermarking for AI-Generated Content",
        "authors": [{"name": "API Fallback Example"}],
        "year": 2024,
        "venue": "Mock Demo Data",
        "abstract": "This is mock demo data used only when Semantic Scholar and arXiv are unavailable.",
        "citationCount": 0,
        "url": "https://example.com/mock-aigc-watermarking",
        "openAccessPdf": {"url": "https://example.com/mock-aigc-watermarking.pdf"},
        "_search_rank": 0,
        "_source": "Mock fallback",
    }
]


def local_keyword_expansion(topic: str, warning: str = "") -> dict[str, Any]:
    keywords = DEFAULT_WATERMARKING_QUERIES
    return {
        "keywords": keywords[:8],
        "search_queries": keywords[:8],
        "search_query": keywords[0],
        "keyword_expansion_source": "local fallback",
        "keyword_expansion_warning": warning,
    }


def expand_keywords_with_deepseek(topic: str) -> dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return local_keyword_expansion(topic, "未检测到 DEEPSEEK_API_KEY，已使用本地英文检索式 fallback。")

    timeout_seconds = float(os.getenv("DEEPSEEK_KEYWORD_TIMEOUT", "6"))
    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=timeout_seconds,
        max_retries=0,
    )
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)

    prompt = f"""
请把下面的中文研究方向扩展成适合论文 API 检索的英文关键词和多个英文检索式。

要求：
1. 只输出 JSON，不要输出解释。
2. 不要生成任何论文标题、作者、年份、venue 或链接。
3. keywords 给出 5-8 个英文关键词或短语。
4. search_queries 给出 5-8 个英文检索式，覆盖相关近义方向。
5. 如果主题与模型水印、AIGC 水印相关，检索式应覆盖：
   AI-generated content watermarking, generative AI watermarking,
   diffusion model watermarking, LLM watermarking, model watermarking,
   neural network watermarking。

研究方向：{topic}

JSON 格式：
{{
  "keywords": ["keyword 1", "keyword 2"],
  "search_queries": ["query 1", "query 2"]
}}
""".strip()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你只负责生成英文检索关键词，不能编造论文列表或论文元数据。",
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        content = response.choices[0].message.content or ""
        data = _parse_json_object(content)
    except Exception as exc:
        return local_keyword_expansion(topic, f"DeepSeek 关键词扩展超时或失败，已使用本地英文检索式 fallback。原因：{exc}")

    keywords = _clean_string_list(data.get("keywords") or [])
    search_queries = _clean_string_list(data.get("search_queries") or [])

    if not search_queries:
        old_query = data.get("search_query")
        search_queries = _clean_string_list([old_query] if old_query else [])
    if not search_queries:
        search_queries = DEFAULT_WATERMARKING_QUERIES
    if not keywords:
        keywords = search_queries

    return {
        "keywords": keywords[:8],
        "search_queries": search_queries[:8],
        "search_query": search_queries[0],
        "keyword_expansion_source": "DeepSeek",
        "keyword_expansion_warning": "",
    }


def _clean_string_list(values: list[Any]) -> list[str]:
    result = []
    for value in values:
        text = str(value).strip().strip('"')
        if text and text not in result:
            result.append(text)
    return result


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def search_semantic_scholar(query: str, limit: int = 20) -> list[dict[str, Any]]:
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    response = requests.get(
        SEMANTIC_SCHOLAR_SEARCH_URL,
        params={
            "query": query,
            "limit": limit,
            "fields": SEMANTIC_SCHOLAR_FIELDS,
        },
        headers=headers,
        timeout=float(os.getenv("PAPER_API_TIMEOUT", DEFAULT_API_TIMEOUT_SECONDS)),
    )
    response.raise_for_status()
    papers = response.json().get("data") or []

    for rank, paper in enumerate(papers):
        paper["_search_rank"] = rank
        paper["_source"] = "Semantic Scholar"
        paper["_query"] = query

    return papers


def search_semantic_scholar_papers(query: str, limit: int = 20) -> list[dict[str, Any]]:
    return search_semantic_scholar(query, limit=limit)


def semantic_scholar_auth_mode() -> str:
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        return "authenticated"
    return "public"


def search_arxiv(query: str, limit: int = 20) -> list[dict[str, Any]]:
    response = requests.get(
        ARXIV_SEARCH_URL,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        timeout=float(os.getenv("PAPER_API_TIMEOUT", DEFAULT_API_TIMEOUT_SECONDS)),
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []

    for rank, entry in enumerate(root.findall("atom:entry", namespace)):
        arxiv_id_url = _xml_text(entry, "atom:id", namespace)
        arxiv_id = arxiv_id_url.rsplit("/", 1)[-1] if arxiv_id_url else ""
        title = re.sub(r"\s+", " ", _xml_text(entry, "atom:title", namespace)).strip()
        summary = re.sub(r"\s+", " ", _xml_text(entry, "atom:summary", namespace)).strip()
        year = _published_year(_xml_text(entry, "atom:published", namespace))
        authors = [
            {"name": _xml_text(author, "atom:name", namespace)}
            for author in entry.findall("atom:author", namespace)
            if _xml_text(author, "atom:name", namespace)
        ]

        papers.append(
            {
                "paperId": f"arxiv:{arxiv_id}" if arxiv_id else "",
                "title": title,
                "authors": authors,
                "year": year,
                "venue": "arXiv",
                "abstract": summary,
                "citationCount": 0,
                "url": arxiv_id_url,
                "openAccessPdf": {"url": f"https://arxiv.org/pdf/{arxiv_id}"} if arxiv_id else None,
                "_search_rank": rank,
                "_source": "arXiv fallback",
                "_query": query,
            }
        )

    return papers


def search_arxiv_papers(query: str, limit: int = 20) -> list[dict[str, Any]]:
    return search_arxiv(query, limit=limit)


def _xml_text(node, path: str, namespace: dict[str, str]) -> str:
    child = node.find(path, namespace)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _published_year(value: str) -> int | None:
    match = re.match(r"(\d{4})", value or "")
    if not match:
        return None
    return int(match.group(1))


def search_dblp_publications(query: str, limit: int = 5) -> list[dict[str, Any]]:
    response = requests.get(
        DBLP_PUBLICATION_SEARCH_URL,
        params={"q": query, "format": "json", "h": limit},
        timeout=float(os.getenv("PAPER_API_TIMEOUT", DEFAULT_API_TIMEOUT_SECONDS)),
    )
    response.raise_for_status()
    hits = (((response.json() or {}).get("result") or {}).get("hits") or {}).get("hit") or []
    if isinstance(hits, dict):
        hits = [hits]

    publications = []
    for hit in hits:
        info = (hit or {}).get("info") or {}
        title = _normalize_dblp_title(info.get("title") or "")
        if not title:
            continue
        publications.append(
            {
                "title": title,
                "authors": _parse_dblp_authors(info.get("authors")),
                "year": _safe_int(info.get("year")),
                "venue": info.get("venue") or "",
                "url": info.get("url") or "",
                "ee": _parse_dblp_ee(info.get("ee")),
            }
        )
    return publications


def _normalize_dblp_title(title: str) -> str:
    title = re.sub(r"<[^>]+>", "", str(title or ""))
    title = re.sub(r"\s+", " ", title).strip()
    return title[:-1].strip() if title.endswith(".") else title


def _parse_dblp_authors(authors_value: Any) -> list[str]:
    if not authors_value:
        return []
    authors = authors_value.get("author") if isinstance(authors_value, dict) else authors_value
    if isinstance(authors, (str, int)):
        authors = [authors]
    if isinstance(authors, dict):
        authors = [authors]

    names = []
    for author in authors or []:
        if isinstance(author, dict):
            name = author.get("text") or author.get("#text") or author.get("name") or ""
        else:
            name = str(author)
        if name:
            names.append(name)
    return names


def _parse_dblp_ee(ee_value: Any) -> str:
    if isinstance(ee_value, list):
        return str(ee_value[0]) if ee_value else ""
    return str(ee_value or "")


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _title_similarity(left: str, right: str) -> float:
    def normalize(value: str) -> str:
        value = re.sub(r"[^a-z0-9 ]+", " ", (value or "").lower())
        return re.sub(r"\s+", " ", value).strip()

    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def find_best_dblp_match(title: str, candidates: list[dict[str, Any]], threshold: float = 0.82) -> dict[str, Any] | None:
    best_match = None
    best_score = 0.0
    for candidate in candidates:
        score = _title_similarity(title, candidate.get("title") or "")
        if score > best_score:
            best_score = score
            best_match = candidate
    if best_match and best_score >= threshold:
        result = dict(best_match)
        result["similarity"] = round(best_score, 3)
        return result
    return None


def enrich_paper_with_dblp(paper: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(paper)
    title = enriched.get("title") or ""
    if enriched.get("_source") == "Mock fallback":
        return _apply_dblp_status(enriched, "not_checked")
    if SOURCE_DBLP in enriched.get("sources", []) and enriched.get("dblp_status") == "found":
        return _apply_dblp_status(enriched, "found")
    if not title:
        return _apply_dblp_status(enriched, "not_checked")

    try:
        candidates = search_dblp_publications(title, limit=5)
        match = find_best_dblp_match(title, candidates)
    except Exception as exc:
        enriched["dblp_error"] = str(exc)
        return _apply_dblp_status(enriched, "error")

    if not match:
        return _apply_dblp_status(enriched, "not_found")

    enriched["dblp_match"] = match
    enriched["dblp_title"] = match.get("title") or ""
    enriched["dblp_authors"] = match.get("authors") or []
    enriched["dblp_year"] = match.get("year")
    enriched["dblp_venue"] = match.get("venue") or ""
    enriched["dblp_url"] = match.get("url") or match.get("ee") or ""
    enriched["dblp_similarity"] = match.get("similarity")
    enriched["quality_verified"] = "正式发表，venue 已确认" if enriched["dblp_venue"] else "正式发表"
    return _apply_dblp_status(enriched, "found")


def verify_with_dblp(paper: dict[str, Any]) -> dict[str, Any]:
    return enrich_paper_with_dblp(paper)


def _apply_dblp_status(paper: dict[str, Any], status: str) -> dict[str, Any]:
    paper["dblp_status"] = status
    paper.setdefault("dblp_venue", "")
    paper.setdefault("dblp_year", None)
    paper.setdefault("dblp_url", "")
    paper["publication_status"] = "confirmed" if status == "found" else "pending"
    paper["formal_publication_status"] = "已确认" if status == "found" else "待确认"
    paper["quality_judgement"] = paper.get("quality_verified") or "待进一步确认"
    paper["dblp_verification"] = {
        "status": status,
        "venue": paper.get("dblp_venue") or "",
        "year": paper.get("dblp_year"),
        "url": paper.get("dblp_url") or "",
        "error": paper.get("dblp_error") or "",
    }
    verification = dict(paper.get("verification") or {})
    verification["dblp_status"] = paper["dblp_status"]
    verification["dblp_venue"] = paper.get("dblp_venue") or ""
    verification["dblp_year"] = paper.get("dblp_year")
    verification["dblp_url"] = paper.get("dblp_url") or ""
    verification["publication_status"] = paper["publication_status"]
    verification["formal_publication_status"] = paper["formal_publication_status"]
    verification["quality_verified"] = paper.get("quality_verified") or verification.get("quality_verified", "")
    verification["quality_judgement"] = paper["quality_judgement"]
    verification["dblp_verification"] = paper["dblp_verification"]
    paper["verification"] = verification
    return paper


def verify_paper(paper: dict[str, Any]) -> dict[str, Any]:
    metadata_checks = {
        "title": bool(paper.get("title")),
        "authors": bool(paper.get("authors")),
        "year": paper.get("year") is not None,
        "traceable_id": bool(paper.get("url") or paper.get("paperId")),
        "abstract": bool(paper.get("abstract")),
        "openAccessPdf": bool(_pdf_url(paper)),
        "citationCount": paper.get("citationCount") is not None,
        "venue": bool(paper.get("venue")),
    }

    if not metadata_checks["title"] or not metadata_checks["traceable_id"]:
        status = "unverified"
    elif all(metadata_checks.values()):
        status = "verified"
    else:
        status = "partial"

    source = paper.get("_source") or SOURCE_SEMANTIC
    source_label = _source_label(source)
    source_verified = "来源可追溯" if source in {"Semantic Scholar", "semantic_scholar", "arXiv fallback", "arxiv", "dblp", "DBLP"} else "mock 示例"
    if source in {"arxiv", "arXiv fallback"} or (paper.get("venue") or "").lower() == "arxiv":
        quality_verified = "arXiv 预印本"
    elif _is_quality_venue(paper.get("venue")):
        quality_verified = "顶会/顶刊"
    else:
        quality_verified = "待进一步确认"
    missing_fields = [field for field, ok in metadata_checks.items() if not ok]

    verified_paper = dict(paper)
    verified_paper["status"] = status
    verified_paper["missing_fields"] = missing_fields
    verified_paper["source_label"] = source_label
    verified_paper["metadata_verified"] = "完整" if status == "verified" else "部分完整"
    verified_paper["source_verified"] = source_verified
    verified_paper["quality_verified"] = quality_verified
    verified_paper["dblp_status"] = paper.get("dblp_status") or "not_checked"
    verified_paper["dblp_venue"] = paper.get("dblp_venue") or ""
    verified_paper["dblp_year"] = paper.get("dblp_year")
    verified_paper["dblp_url"] = paper.get("dblp_url") or ""
    verified_paper["publication_status"] = paper.get("publication_status") or "pending"
    verified_paper["formal_publication_status"] = "待确认"
    if verified_paper["publication_status"] == "confirmed":
        verified_paper["formal_publication_status"] = "已确认"
    verified_paper["quality_judgement"] = paper.get("quality_judgement") or quality_verified
    verified_paper["dblp_verification"] = {
        "status": verified_paper["dblp_status"],
        "venue": "",
        "year": None,
        "url": "",
        "error": "",
    }
    verified_paper["verification"] = {
        "status": status,
        "metadata_verified": verified_paper["metadata_verified"],
        "source_verified": source_verified,
        "quality_verified": quality_verified,
        "dblp_status": verified_paper["dblp_status"],
        "dblp_venue": verified_paper["dblp_venue"],
        "dblp_year": verified_paper["dblp_year"],
        "dblp_url": verified_paper["dblp_url"],
        "publication_status": verified_paper["publication_status"],
        "formal_publication_status": verified_paper["formal_publication_status"],
        "quality_judgement": verified_paper["quality_judgement"],
        "dblp_verification": verified_paper["dblp_verification"],
        "missing_fields": missing_fields,
        "source": source,
        "source_label": source_label,
    }
    return verified_paper


def _source_label(source: str) -> str:
    if source in {"arxiv", "arXiv fallback"}:
        return "arXiv"
    if source in {"mock", "Mock fallback"}:
        return "mock"
    if source == "dblp":
        return "DBLP"
    return "Semantic Scholar"


def _is_quality_venue(venue: str | None) -> bool:
    if not venue:
        return False
    normalized = venue.lower()
    if normalized == "arxiv":
        return False
    return any(good_venue in normalized for good_venue in QUALITY_VENUES)


def filter_verified_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified = []
    for paper in papers:
        checked = verify_paper(paper)
        if checked["status"] in {"verified", "partial"}:
            verified.append(checked)
    return verified


def normalize_semantic_scholar_paper(paper: dict[str, Any]) -> dict[str, Any]:
    pdf_url = _pdf_url(paper)
    normalized = {
        "title": paper.get("title") or "",
        "authors": paper.get("authors") or [],
        "year": paper.get("year"),
        "venue": paper.get("venue") or "",
        "abstract": paper.get("abstract") or "",
        "citation_count": paper.get("citationCount"),
        "citationCount": paper.get("citationCount"),
        "url": paper.get("url") or "",
        "pdf_url": pdf_url,
        "openAccessPdf": {"url": pdf_url} if pdf_url else None,
        "doi": _extract_doi(paper),
        "arxiv_id": _extract_arxiv_id(paper),
        "paper_id": paper.get("paperId") or "",
        "paperId": paper.get("paperId") or "",
        "sources": [SOURCE_SEMANTIC],
        "source_metadata": {SOURCE_SEMANTIC: paper},
        "publication_status": "pending",
        "metadata_completeness": "",
        "quality_judgement": "",
        "_source": SOURCE_SEMANTIC,
        "_search_rank": paper.get("_search_rank", 9999),
        "_query": paper.get("_query", ""),
    }
    return normalized


def normalize_arxiv_paper(paper: dict[str, Any]) -> dict[str, Any]:
    pdf_url = _pdf_url(paper)
    normalized = {
        "title": paper.get("title") or "",
        "authors": paper.get("authors") or [],
        "year": paper.get("year"),
        "venue": "arXiv",
        "abstract": paper.get("abstract") or "",
        "citation_count": None,
        "citationCount": None,
        "url": paper.get("url") or "",
        "pdf_url": pdf_url,
        "openAccessPdf": {"url": pdf_url} if pdf_url else None,
        "doi": "",
        "arxiv_id": _extract_arxiv_id(paper),
        "paper_id": paper.get("paperId") or "",
        "paperId": paper.get("paperId") or "",
        "sources": [SOURCE_ARXIV],
        "source_metadata": {SOURCE_ARXIV: paper},
        "publication_status": "pending",
        "metadata_completeness": "",
        "quality_judgement": "arXiv 预印本",
        "_source": "arXiv fallback",
        "_search_rank": paper.get("_search_rank", 9999),
        "_query": paper.get("_query", ""),
    }
    return normalized


def normalize_dblp_paper(paper: dict[str, Any]) -> dict[str, Any]:
    venue = paper.get("venue") or ""
    url = paper.get("url") or paper.get("ee") or ""
    publication_status = "confirmed" if _is_formal_dblp_venue(venue) else "pending"
    normalized = {
        "title": paper.get("title") or "",
        "authors": [{"name": author} for author in paper.get("authors") or []],
        "year": paper.get("year"),
        "venue": venue,
        "abstract": "",
        "citation_count": None,
        "citationCount": None,
        "url": url,
        "pdf_url": "",
        "openAccessPdf": None,
        "doi": _extract_doi(paper),
        "arxiv_id": _extract_arxiv_id(paper),
        "paper_id": "",
        "paperId": "",
        "sources": [SOURCE_DBLP],
        "source_metadata": {SOURCE_DBLP: paper},
        "publication_status": publication_status,
        "metadata_completeness": "",
        "quality_judgement": "正式发表，venue 已确认" if publication_status == "confirmed" else "待进一步确认",
        "dblp_status": "found",
        "dblp_title": paper.get("title") or "",
        "dblp_authors": paper.get("authors") or [],
        "dblp_year": paper.get("year"),
        "dblp_venue": venue,
        "dblp_url": url,
        "dblp_verification": {
            "status": "found",
            "venue": venue,
            "year": paper.get("year"),
            "url": url,
            "error": "",
        },
        "_source": SOURCE_DBLP,
        "_search_rank": paper.get("_search_rank", 9999),
        "_query": paper.get("_query", ""),
    }
    return normalized


def merge_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for paper in papers:
        match = _find_merge_match(merged, paper)
        if match is None:
            merged.append(dict(paper))
        else:
            _merge_paper_into(match, paper)
    return merged


def _find_merge_match(existing_papers: list[dict[str, Any]], paper: dict[str, Any]) -> dict[str, Any] | None:
    for existing in existing_papers:
        if paper.get("doi") and existing.get("doi") and paper["doi"].lower() == existing["doi"].lower():
            return existing
        if paper.get("arxiv_id") and existing.get("arxiv_id") and paper["arxiv_id"] == existing["arxiv_id"]:
            return existing
        left_title = _title_key(existing.get("title"))
        right_title = _title_key(paper.get("title"))
        if left_title and right_title and left_title == right_title:
            return existing
        years_close = existing.get("year") and paper.get("year") and abs(existing["year"] - paper["year"]) <= 1
        if years_close and _title_similarity(existing.get("title") or "", paper.get("title") or "") >= 0.9:
            return existing
    return None


def _merge_paper_into(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for source in incoming.get("sources") or []:
        if source not in target.setdefault("sources", []):
            target["sources"].append(source)
    target.setdefault("source_metadata", {}).update(incoming.get("source_metadata") or {})

    if not target.get("abstract") or (SOURCE_SEMANTIC in incoming.get("sources", []) and incoming.get("abstract")):
        target["abstract"] = incoming.get("abstract") or target.get("abstract", "")
    if incoming.get("citation_count") is not None:
        target["citation_count"] = incoming.get("citation_count")
        target["citationCount"] = incoming.get("citation_count")
    if not target.get("pdf_url") or SOURCE_ARXIV in incoming.get("sources", []):
        target["pdf_url"] = incoming.get("pdf_url") or target.get("pdf_url", "")
        target["openAccessPdf"] = {"url": target["pdf_url"]} if target.get("pdf_url") else target.get("openAccessPdf")
    if _is_formal_dblp_venue(incoming.get("venue")):
        target["venue"] = incoming.get("venue") or target.get("venue", "")
        target["publication_status"] = "confirmed"
        target["quality_judgement"] = "正式发表，venue 已确认"
        target["dblp_status"] = "found"
        target["dblp_venue"] = incoming.get("venue") or ""
        target["dblp_year"] = incoming.get("year")
        target["dblp_url"] = incoming.get("url") or ""
    for field in ["doi", "arxiv_id", "paper_id", "paperId", "url", "year"]:
        if not target.get(field) and incoming.get(field):
            target[field] = incoming[field]
    if not target.get("authors") and incoming.get("authors"):
        target["authors"] = incoming["authors"]


def _title_key(title: str | None) -> str:
    return re.sub(r"\s+", " ", (title or "").lower()).strip()


def _extract_doi(paper: dict[str, Any]) -> str:
    candidates = [paper.get("doi"), paper.get("externalIds", {}).get("DOI") if isinstance(paper.get("externalIds"), dict) else "", paper.get("ee"), paper.get("url")]
    for value in candidates:
        match = re.search(r"10\.\d{4,9}/[^\s]+", str(value or ""), flags=re.I)
        if match:
            return match.group(0).rstrip(".")
    return ""


def _extract_arxiv_id(paper: dict[str, Any]) -> str:
    values = [paper.get("paperId"), paper.get("url"), paper.get("pdf_url"), _pdf_url(paper)]
    for value in values:
        text = str(value or "")
        match = re.search(r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5}(?:v\d+)?)", text, flags=re.I)
        if match:
            return match.group(1)
    return ""


def _is_formal_dblp_venue(venue: str | None) -> bool:
    normalized = (venue or "").strip().lower()
    return bool(normalized and normalized not in {"corr", "arxiv"})


def merge_and_deduplicate_by_title(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []

    for paper in papers:
        title_key = re.sub(r"\s+", " ", (paper.get("title") or "").lower()).strip()
        if not title_key or title_key in seen:
            continue
        seen.add(title_key)
        unique.append(paper)

    return unique


def deduplicate_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return merge_and_deduplicate_by_title(papers)


def score_paper(paper: dict[str, Any], keywords: list[str] | None = None) -> float:
    keywords = keywords or []
    score = 0.0
    current_year = date.today().year
    year = paper.get("year") or 0

    if year >= current_year - 2:
        score += 3
    elif year >= current_year - 5:
        score += 1

    hits = hit_keywords(paper, keywords)
    score += min(6, len(hits) * 2)

    if _pdf_url(paper):
        score += 2

    citation_count = paper.get("citationCount") or 0
    if citation_count > 0:
        score += min(3, math.log10(citation_count + 1))

    if _is_quality_venue(paper.get("venue")):
        score += 3

    source = paper.get("_source")
    if source == "arXiv fallback":
        score += 0.5

    return round(score, 3)


def sort_papers(papers: list[dict[str, Any]], keywords: list[str] | None = None) -> list[dict[str, Any]]:
    for paper in papers:
        paper["hit_keywords"] = hit_keywords(paper, keywords or [])
        paper["score"] = score_paper(paper, keywords)

    return sorted(
        papers,
        key=lambda paper: (
            -paper.get("score", 0),
            paper.get("_search_rank", 9999),
            -(paper.get("year") or 0),
            -(paper.get("citationCount") or 0),
        ),
    )


def discover_papers(topic: str, limit: int = 10, selected_sources: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    expanded = expand_keywords_with_deepseek(topic)
    search_queries = expanded.get("search_queries") or [expanded.get("search_query") or topic]
    keywords = expanded.get("keywords") or search_queries
    selected_sources = normalize_selected_sources(selected_sources or DEFAULT_SELECTED_SOURCES)
    raw_papers: list[dict[str, Any]] = []
    debug_info: dict[str, Any] = {
        "user_query": topic,
        "expanded_keywords": keywords,
        "selected_sources": selected_sources,
        "sources": {
            SOURCE_ARXIV: {"called": False, "count": 0, "error": ""},
            SOURCE_SEMANTIC: {"called": False, "count": 0, "error": ""},
            SOURCE_DBLP: {"called": False, "count": 0, "error": ""},
        },
        "pre_merge_count": 0,
        "deduplicated_count": 0,
        "mock_fallback": False,
        "mock_fallback_reason": "",
    }
    warnings = []
    failed_sources = set()

    def collect_source(source_name: str, search_fn, normalize_fn) -> None:
        source_results = []
        debug_info["sources"][source_name]["called"] = True
        try:
            for query in search_queries:
                if not query:
                    continue
                found = search_fn(query, limit=max(limit, 10))
                source_results.extend(found)
                if len(source_results) >= limit * 2:
                    break
            normalized = [normalize_fn(paper) for paper in source_results]
            raw_papers.extend(normalized)
            debug_info["sources"][source_name]["count"] = len(normalized)
            debug_info["sources"][source_name]["error"] = ""
        except Exception as exc:
            failed_sources.add(source_name)
            debug_info["sources"][source_name]["count"] = 0
            debug_info["sources"][source_name]["error"] = str(exc)
            if source_name == SOURCE_SEMANTIC and "429" in str(exc):
                warnings.append("Semantic Scholar 未认证模式受到限流，已切换到 arXiv fallback。")
            else:
                warnings.append(f"{source_name} 请求失败：{exc}")

    if SOURCE_ARXIV in selected_sources:
        collect_source(SOURCE_ARXIV, search_arxiv_papers, normalize_arxiv_paper)
    if SOURCE_SEMANTIC in selected_sources:
        collect_source(SOURCE_SEMANTIC, search_semantic_scholar_papers, normalize_semantic_scholar_paper)
        if semantic_scholar_auth_mode() == "public" and SOURCE_SEMANTIC not in failed_sources:
            warnings.append("当前使用 Semantic Scholar 未认证模式，可能受到公共限流影响。")
    if SOURCE_DBLP in selected_sources:
        collect_source(SOURCE_DBLP, search_dblp_publications, normalize_dblp_paper)

    real_sources = [source for source in selected_sources if source in {SOURCE_ARXIV, SOURCE_SEMANTIC, SOURCE_DBLP}]
    successful_count = sum(debug_info["sources"][source]["count"] for source in real_sources)
    all_failed = bool(real_sources) and failed_sources.issuperset(real_sources)
    all_empty = bool(real_sources) and successful_count == 0
    if all_failed or all_empty:
        reason = "所有真实数据源请求失败" if all_failed else "所有真实数据源返回 0 条"
        raw_papers = [
            {
                **normalize_semantic_scholar_paper(paper),
                "_source": "Mock fallback",
                "sources": ["mock"],
                "source_metadata": {"mock": paper},
            }
            for paper in MOCK_PAPERS
        ]
        debug_info["mock_fallback"] = True
        debug_info["mock_fallback_reason"] = reason
        warnings.append(f"{reason}，当前展示 mock fallback 示例数据。")

    debug_info["pre_merge_count"] = len(raw_papers)
    merged_papers = merge_papers(raw_papers)
    checked_papers = filter_verified_papers(merged_papers)
    unique_papers = merge_and_deduplicate_by_title(checked_papers)
    debug_info["deduplicated_count"] = len(unique_papers)
    sorted_results = sort_papers(unique_papers, keywords)
    if SOURCE_DBLP in selected_sources:
        display_results = [enrich_paper_with_dblp(paper) for paper in sorted_results[:limit]]
    else:
        display_results = sorted_results[:limit]
    for paper in display_results:
        paper["metadata_completeness"] = paper.get("metadata_verified", "")
        paper["recommendation_reason"] = generate_recommendation_reason(topic, paper)
        paper["reading_advice"] = generate_reading_advice(paper)
        paper["recommendation_priority"] = calculate_recommendation_priority(paper)

    successful_sources = sorted({source for paper in display_results for source in paper.get("sources", []) if source != "mock"})
    if debug_info.get("mock_fallback"):
        source = "mock fallback"
    elif SOURCE_SEMANTIC in failed_sources and SOURCE_ARXIV in selected_sources and debug_info["sources"][SOURCE_ARXIV]["count"] > 0:
        source = "arXiv fallback"
    else:
        source = " + ".join(successful_sources or selected_sources)
    warning = "；".join(warnings)

    return {
        "topic": topic,
        "keywords": keywords,
        "search_queries": search_queries,
        "search_query": search_queries[0],
        "keyword_expansion_source": expanded.get("keyword_expansion_source", "DeepSeek"),
        "keyword_expansion_warning": expanded.get("keyword_expansion_warning", ""),
        "source": source,
        "selected_sources": selected_sources,
        "semantic_scholar_auth_mode": semantic_scholar_auth_mode(),
        "warning": warning,
        "source_warning": build_source_warning(source, semantic_scholar_auth_mode(), warning),
        "debug_info": debug_info,
        "papers": display_results,
    }


def normalize_selected_sources(values: list[str] | tuple[str, ...]) -> list[str]:
    aliases = {
        "arxiv": SOURCE_ARXIV,
        "arXiv": SOURCE_ARXIV,
        "semantic_scholar": SOURCE_SEMANTIC,
        "Semantic Scholar": SOURCE_SEMANTIC,
        "SemanticScholar": SOURCE_SEMANTIC,
        "semantic": SOURCE_SEMANTIC,
        "dblp": SOURCE_DBLP,
        "DBLP": SOURCE_DBLP,
        "dblp_search": SOURCE_DBLP,
    }
    normalized = []
    for value in values:
        source = aliases.get(str(value), str(value).lower())
        if source in {SOURCE_ARXIV, SOURCE_SEMANTIC, SOURCE_DBLP} and source not in normalized:
            normalized.append(source)
    return normalized


def build_source_warning(source: str, auth_mode: str, warning: str = "") -> str:
    if source == "arXiv fallback" and auth_mode == "public":
        return "当前 Semantic Scholar 未配置 API Key，系统已尝试未认证检索；由于公共限流，已自动切换到 arXiv 预印本数据源。"
    return warning


def determine_search_status(result: dict[str, Any] | None) -> str:
    if not result:
        return "failed"

    source = result.get("source") or ""
    papers = result.get("papers") or []
    if source == "mock fallback":
        return "fallback_mock"
    if papers:
        return "success"
    return "failed"


def can_reuse_discovery_result(
    current_query: str,
    last_query: str,
    last_result: dict[str, Any] | None,
    last_search_status: str,
    last_source: str,
    force_refresh: bool = False,
) -> bool:
    if force_refresh:
        return False
    if current_query != last_query:
        return False
    if not last_result:
        return False
    if last_search_status != "success":
        return False
    if last_source == "mock fallback":
        return False

    papers = last_result.get("papers") or []
    return any((paper.get("_source") or "") != "Mock fallback" for paper in papers)


def calculate_recommendation_priority(paper: dict[str, Any]) -> str:
    score = paper.get("score") or 0
    hit_count = len(paper.get("hit_keywords") or [])
    year = paper.get("year") or 0
    citation_count = paper.get("citationCount") or 0
    has_pdf = bool(_pdf_url(paper))
    current_year = date.today().year

    evidence_score = 0
    if score >= 9:
        evidence_score += 2
    elif score >= 5:
        evidence_score += 1
    if hit_count:
        evidence_score += 1
    if year >= current_year - 2:
        evidence_score += 1
    if citation_count >= 50:
        evidence_score += 1
    if has_pdf:
        evidence_score += 1
    if _is_quality_venue(paper.get("venue")):
        evidence_score += 1

    if evidence_score >= 5:
        return "高"
    if evidence_score >= 2:
        return "中"
    return "低"


def generate_recommendation_reason(topic: str, paper: dict[str, Any]) -> str:
    fallback = _fallback_recommendation_reason(topic, paper)
    if os.getenv("DEEPSEEK_REASON_ENABLE", "0") != "1":
        return fallback

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return fallback

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=8.0, max_retries=0)
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    metadata = {
        "topic": topic,
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "venue": paper.get("venue"),
        "year": paper.get("year"),
        "citationCount": paper.get("citationCount"),
        "source": paper.get("_source"),
    }
    prompt = f"""
请基于下面 API 返回的真实论文元数据，给出 1-2 句中文推荐理由。

严格要求：
1. 只能使用给定元数据。
2. 不要编造作者、会议、年份、标题或链接。
3. 如果来源是 arXiv，只能称为预印本，不要说它是顶会或顶刊。

元数据 JSON：
{json.dumps(metadata, ensure_ascii=False)}
""".strip()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你只基于 API 元数据写推荐理由，不能补全或编造论文信息。"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        content = response.choices[0].message.content or ""
        return content.strip() or fallback
    except Exception:
        return fallback


def _fallback_recommendation_reason(topic: str, paper: dict[str, Any]) -> str:
    year = paper.get("year") or "年份未知"
    venue = paper.get("venue") or "来源 venue 未提供"
    source = paper.get("_source") or "API"
    if venue == "arXiv":
        return f"该预印本与“{topic}”相关，年份为 {year}，可作为了解最新研究线索的参考；其质量仍需进一步核查。"
    return f"该论文与“{topic}”相关，来源为 {source}，年份为 {year}，venue 为 {venue}，可作为候选阅读文献。"


def hit_keywords(paper: dict[str, Any], keywords: list[str]) -> list[str]:
    searchable_text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".lower()
    hits = []
    for keyword in keywords:
        normalized_keyword = keyword.lower().strip()
        if normalized_keyword and normalized_keyword in searchable_text:
            hits.append(keyword)
    return hits


def generate_reading_advice(paper: dict[str, Any]) -> str:
    source = paper.get("_source")
    venue = paper.get("venue") or ""
    citation_count = paper.get("citationCount") or 0
    abstract = paper.get("abstract") or ""

    if source == "arXiv fallback" or venue.lower() == "arxiv":
        return "适合追踪最新预印本，正式发表 venue 需要后续通过 Semantic Scholar、DBLP 或 Crossref 进一步核查。"
    if _is_quality_venue(venue):
        return "建议精读方法部分和实验设置，可作为重点候选文献。"
    if citation_count >= 100:
        return "引用数较高，适合作为综述背景阅读。"
    if len(abstract) > 500:
        return "摘要信息较完整，建议先快速浏览摘要和结论，再决定是否精读。"
    return "可作为补充线索阅读，建议结合原文方法和实验部分进一步判断价值。"


def format_authors(paper: dict[str, Any], max_authors: int = 6) -> str:
    authors = paper.get("authors") or []
    names = [author.get("name", "") for author in authors if author.get("name")]
    if not names:
        return "API 未提供"
    if len(names) > max_authors:
        return ", ".join(names[:max_authors]) + " 等"
    return ", ".join(names)


def paper_pdf_url(paper: dict[str, Any]) -> str:
    return _pdf_url(paper) or ""


def _pdf_url(paper: dict[str, Any]) -> str:
    if paper.get("pdf_url"):
        return paper.get("pdf_url") or ""
    open_access_pdf = paper.get("openAccessPdf")
    if isinstance(open_access_pdf, dict):
        return open_access_pdf.get("url") or ""
    return ""
