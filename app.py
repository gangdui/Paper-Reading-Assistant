from collections import Counter
from difflib import SequenceMatcher
from io import BytesIO
import os
import re

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APIStatusError, OpenAI
import pdfplumber
import requests
import streamlit as st

from paper_discovery import (
    can_reuse_discovery_result,
    determine_search_status,
    discover_papers,
    format_authors,
    paper_pdf_url,
    search_arxiv_papers,
    search_semantic_scholar_papers,
)


MAX_AI_INPUT_CHARS = 15000
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "experiment",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
}

SAMPLE_MARKDOWN_NOTES = """# 结构化文献笔记（示例）

## 一句话总结
本文展示了一个面向研究生论文阅读的结构化笔记生成流程，用于帮助用户快速把握论文主线。

## 论文摘要
这是课堂演示用示例输出，不基于真实论文内容。

## 研究背景与研究问题
原文未明确说明

## 主要内容概述
示例展示了系统如何按照固定模块组织文献笔记。

## 提出的方法
原文未明确说明

## 实验设计与结果分析
原文未明确说明

## 主要贡献
示例说明了结构化文献笔记的展示格式，便于课堂演示。

## 不足与局限
该内容为示例输出，不代表真实论文分析结果。

## 研究生阅读建议
建议先阅读摘要、引言和结论，再结合方法与实验部分做重点标注。
"""


def extract_text_from_pdf(pdf_file, progress_bar=None, status_text=None) -> str:
    """Extract text from all pages of an uploaded PDF file."""
    pdf_bytes = pdf_file.read()
    text_parts = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        total_pages = max(len(pdf.pages), 1)
        for index, page in enumerate(pdf.pages, start=1):
            if status_text is not None:
                status_text.info(f"正在读取 PDF：第 {index} / {total_pages} 页")
            if progress_bar is not None:
                percent = int(index / total_pages * 100)
                progress_bar.progress(percent, text=f"PDF 读取进度：{index} / {total_pages} 页")

            page_text = page.extract_text() or ""
            text_parts.append(page_text)

    return "\n\n".join(text_parts).strip()


def is_section_heading(line: str) -> bool:
    normalized = re.sub(r"^\d+(\.\d+)*\s+", "", line.strip().lower())
    normalized = normalized.strip(":.- ")
    return normalized in SECTION_HEADINGS


def is_page_number(line: str) -> bool:
    line = line.strip()
    return bool(
        re.fullmatch(r"\d+", line)
        or re.fullmatch(r"-\s*\d+\s*-", line)
        or re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", line, flags=re.IGNORECASE)
    )


def clean_paper_text(text: str) -> str:
    """Clean common PDF extraction noise while keeping paper structure readable."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    non_empty_lines = [line for line in raw_lines if line]
    line_counts = Counter(non_empty_lines)

    filtered_lines = []
    for line in raw_lines:
        if not line:
            filtered_lines.append("")
            continue

        repeated_noise = line_counts[line] >= 3 and len(line) <= 120
        if is_page_number(line):
            continue
        if repeated_noise and not is_section_heading(line):
            continue

        filtered_lines.append(line)

    paragraphs = []
    current = []

    for line in filtered_lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue

        if is_section_heading(line):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(line)
            continue

        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    cleaned = "\n\n".join(paragraphs)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_note_prompt(paper_text: str) -> str:
    return f"""
请基于下面的论文文本，生成中文结构化文献笔记。

要求：
1. 必须严格包含以下 9 个模块，模块标题不能省略：
   - 一句话总结
   - 论文摘要
   - 研究背景与研究问题
   - 主要内容概述
   - 提出的方法
   - 实验设计与结果分析
   - 主要贡献
   - 不足与局限
   - 研究生阅读建议
2. 只根据原文内容总结，不要编造论文中没有的信息。
3. 如果原文没有明确说明某部分内容，请在对应模块输出“原文未明确说明”。
4. 表达要适合研究生阅读，清晰、准确、简洁。
5. 请使用 Markdown 格式输出。

论文文本：
{paper_text}
""".strip()


def generate_structured_notes(cleaned_text: str) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未检测到 DEEPSEEK_API_KEY。请在 .env 文件中配置后重启 Streamlit。")

    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=60.0)
    prompt = build_note_prompt(cleaned_text[:MAX_AI_INPUT_CHARS])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个严谨的研究生论文阅读助手。你必须用中文回答，并且不能编造原文未说明的信息。",
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek 返回了空结果，请稍后重试。")
    return content.strip()


def paper_state_key(paper: dict) -> str:
    raw_key = paper.get("paperId") or paper.get("url") or paper.get("title") or "unknown-paper"
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_key)[:80]


def add_or_update_library_paper(paper: dict, status: str) -> None:
    key = paper_state_key(paper)
    item = dict(st.session_state.library_papers.get(key, {}))
    item.update(
        {
            "key": key,
            "title": paper.get("title") or "API 未提供标题",
            "year": paper.get("year"),
            "venue": paper.get("venue") or "",
            "source": paper_display_source(paper),
            "url": paper.get("url") or "",
            "pdf_url": paper_pdf_url(paper),
            "status": status,
            "has_full_notes": key in st.session_state.full_reading_notes,
        }
    )
    st.session_state.library_papers[key] = item


def update_library_note_status(paper_key: str) -> None:
    if paper_key in st.session_state.library_papers:
        st.session_state.library_papers[paper_key]["has_full_notes"] = paper_key in st.session_state.full_reading_notes


def library_markdown() -> str:
    lines = ["# 我的文献库", ""]
    grouped = group_library_by_status()
    for status in ["待读", "正在读", "精读", "已读", "不相关", "未读"]:
        papers = grouped.get(status, [])
        if not papers:
            continue
        lines.append(f"## {status}")
        for paper in papers:
            note_text = "已有笔记" if paper.get("has_full_notes") else "暂无笔记"
            link = paper.get("url") or ""
            link_text = f" [论文链接]({link})" if link else ""
            lines.append(
                f"- {paper.get('title')}（{paper.get('year') or '年份未提供'}，"
                f"{paper.get('venue') or 'venue 未提供'}，{paper.get('source') or '来源未提供'}，{note_text}）{link_text}"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def group_library_by_status() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for paper in st.session_state.library_papers.values():
        grouped.setdefault(paper.get("status") or "未读", []).append(paper)
    return grouped


def build_quick_reading_prompt(paper: dict, topic: str) -> str:
    metadata = {
        "research_topic": topic,
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "source": paper.get("_source"),
        "keywords": paper.get("hit_keywords") or [],
    }
    return f"""
请基于下面 API 返回的论文元数据，生成中文快速解读。

严格要求：
1. 只能使用给定元数据，不要编造作者、年份、会议、论文标题、链接或全文内容。
2. 这是基于标题和摘要的快速解读，不代表全文阅读。
3. 如果摘要不足以判断，请明确写“摘要信息不足，需阅读全文确认”。
4. 请使用 Markdown 输出，并包含以下小标题：
   - 基于摘要的提示
   - 这篇论文主要研究什么
   - 为什么和当前研究方向相关
   - 可能的方法或研究路线
   - 适合精读还是泛读
   - 建议重点阅读哪些部分

论文元数据：
{metadata}
""".strip()


def generate_quick_reading(paper: dict, topic: str) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未检测到 DEEPSEEK_API_KEY。请在 .env 文件中配置后重启 Streamlit。")

    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=45.0)
    prompt = build_quick_reading_prompt(paper, topic)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是严谨的论文阅读助手。你只能基于用户提供的论文元数据做摘要解读，不能补充或编造未给出的论文信息。",
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek 返回了空结果，请稍后重试。")
    return content.strip()


def download_pdf_bytes(pdf_url: str, timeout: int = 30) -> bytes:
    response = requests.get(pdf_url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
        raise ValueError("下载链接未返回 PDF 内容。")
    return response.content


def ensure_fulltext_for_paper(paper: dict) -> dict:
    existing_pdf = paper_pdf_url(paper)
    debug = {
        "executed": True,
        "has_existing_pdf_url": bool(existing_pdf),
        "has_existing_arxiv_id": bool(paper.get("arxiv_id")),
        "arxiv_id_fields_checked": [],
        "arxiv_id_from_metadata": "",
        "session_state_updated": False,
        "failure_step": "",
        "semantic_scholar_success": False,
        "semantic_scholar_found_pdf": False,
        "semantic_scholar_queries": [],
        "arxiv_queries": [],
        "arxiv_candidate_count": 0,
        "match_reason": "",
        "pdf_url": "",
        "errors": [],
    }
    if existing_pdf:
        debug["pdf_url"] = existing_pdf
        enriched = dict(paper)
        enriched.setdefault("fulltext_status", "found")
        return {"status": "fulltext_available", "pdf_url": existing_pdf, "paper": enriched, "message": "已找到 PDF，正在生成全文文献笔记。", "debug": debug}

    key = paper_state_key(paper)
    cached_lookup = st.session_state.fulltext_lookup_cache.get(key)
    if cached_lookup and cached_lookup.get("status") != "fulltext_not_found":
        return cached_lookup

    enriched = dict(paper)

    arxiv_id, arxiv_source = find_arxiv_id_in_paper(paper)
    debug["arxiv_id_fields_checked"] = arxiv_fulltext_id_values(paper)
    debug["arxiv_id_from_metadata"] = arxiv_id
    if arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        enriched = apply_arxiv_fulltext(enriched, arxiv_id, pdf_url, "found_by_arxiv_id")
        debug["match_reason"] = f"从 {arxiv_source} 提取到 arXiv ID"
        debug["pdf_url"] = pdf_url
        result = {
            "status": "fulltext_found_by_enrichment",
            "pdf_url": pdf_url,
            "paper": enriched,
            "message": "已通过 arXiv 找到开放 PDF，正在生成全文文献笔记。",
            "debug": debug,
        }
        st.session_state.fulltext_lookup_cache[key] = result
        return result

    semantic_queries = semantic_fulltext_queries(paper)
    arxiv_queries = arxiv_fulltext_queries(paper)

    for query in semantic_queries:
        debug["semantic_scholar_queries"].append(query)
        try:
            candidates = search_semantic_scholar_papers(query, limit=5)
            debug["semantic_scholar_success"] = True
            for candidate in candidates:
                pdf_url = paper_pdf_url(candidate)
                matched, reason = fulltext_candidate_match_reason(paper, candidate)
                if pdf_url and matched:
                    enriched = apply_arxiv_fulltext(enriched, extract_arxiv_id_from_paper(candidate), pdf_url, "found_by_semantic_scholar")
                    enriched["fulltext_status"] = "found_by_semantic_scholar"
                    enriched.setdefault("source_metadata", {})["fulltext_semantic_scholar"] = candidate
                    if "semantic_scholar" not in enriched.setdefault("sources", []):
                        enriched["sources"].append("semantic_scholar")
                    debug["semantic_scholar_found_pdf"] = True
                    debug["match_reason"] = f"Semantic Scholar: {reason}"
                    debug["pdf_url"] = pdf_url
                    result = {
                        "status": "fulltext_found_by_enrichment",
                        "pdf_url": pdf_url,
                        "paper": enriched,
                        "message": "已找到开放 PDF，正在生成全文文献笔记。",
                        "debug": debug,
                    }
                    st.session_state.fulltext_lookup_cache[key] = result
                    return result
        except Exception as exc:
            debug["errors"].append(f"Semantic Scholar: {exc}")

    for query in arxiv_queries:
        debug["arxiv_queries"].append(query)
        try:
            candidates = search_arxiv_papers(query, limit=8)
            debug["arxiv_candidate_count"] += len(candidates)
            for candidate in candidates:
                pdf_url = paper_pdf_url(candidate)
                matched, reason = fulltext_candidate_match_reason(paper, candidate)
                if pdf_url and matched:
                    arxiv_id = candidate.get("arxiv_id") or extract_arxiv_id_from_paper(candidate) or enriched.get("arxiv_id", "")
                    enriched = apply_arxiv_fulltext(enriched, arxiv_id, pdf_url, "found_by_arxiv")
                    enriched["url"] = candidate.get("url") or enriched.get("url", "")
                    enriched.setdefault("source_metadata", {})["arxiv"] = candidate
                    debug["match_reason"] = f"arXiv: {reason}"
                    debug["pdf_url"] = pdf_url
                    result = {
                        "status": "fulltext_found_by_enrichment",
                        "pdf_url": pdf_url,
                        "paper": enriched,
                        "message": "已通过 arXiv 找到开放 PDF，正在生成全文文献笔记。",
                        "debug": debug,
                    }
                    st.session_state.fulltext_lookup_cache[key] = result
                    return result
        except Exception as exc:
            debug["errors"].append(f"arXiv: {exc}")

    result = {
        "status": "fulltext_not_found",
        "pdf_url": "",
        "paper": {**enriched, "fulltext_status": "not_found"},
        "message": "未找到开放全文，请切换到“阅读 PDF”页面手动上传论文。",
        "error": "；".join(debug["errors"]),
        "debug": debug,
    }
    debug["failure_step"] = "未从已有字段、Semantic Scholar 或 arXiv 标题搜索找到开放 PDF"
    st.session_state.fulltext_lookup_cache[key] = result
    return result


def fulltext_lookup_queries(paper: dict) -> list[str]:
    values = [
        paper.get("doi"),
        paper.get("arxiv_id"),
        paper.get("paper_id"),
        paper.get("paperId"),
        paper.get("dblp_url"),
        paper.get("title"),
    ]
    return [str(value).strip() for value in values if str(value or "").strip()]


def extract_arxiv_id_from_text(value) -> str:
    text = str(value or "")
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?",
        r"arxiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?",
        r"10\.48550/arxiv\.(\d{4}\.\d{4,5})(?:v\d+)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1)
    return ""


def arxiv_fulltext_id_values(paper: dict) -> list[dict]:
    values = []
    for field in ["arxiv_id", "url", "doi", "paper_id", "paperId", "dblp_url", "dblp_ee"]:
        if paper.get(field):
            values.append({"field": field, "value": paper.get(field)})

    source_metadata = paper.get("source_metadata") or {}
    for source_name, metadata in source_metadata.items():
        if isinstance(metadata, dict):
            for field in ["arxiv_id", "url", "doi", "ee", "dblp_url", "dblp_ee", "paperId"]:
                if metadata.get(field):
                    values.append({"field": f"source_metadata.{source_name}.{field}", "value": metadata.get(field)})
            info = metadata.get("info")
            if isinstance(info, dict):
                for field in ["ee", "url", "doi"]:
                    if info.get(field):
                        values.append({"field": f"source_metadata.{source_name}.info.{field}", "value": info.get(field)})
    return values


def find_arxiv_id_in_paper(paper: dict) -> tuple[str, str]:
    for item in arxiv_fulltext_id_values(paper):
        arxiv_id = extract_arxiv_id_from_text(item["value"])
        if arxiv_id:
            return arxiv_id, item["field"]
    return "", ""


def apply_arxiv_fulltext(paper: dict, arxiv_id: str, pdf_url: str, status: str) -> dict:
    enriched = dict(paper)
    if arxiv_id:
        enriched["arxiv_id"] = arxiv_id
        enriched["url"] = f"https://arxiv.org/abs/{arxiv_id}"
    enriched["pdf_url"] = pdf_url
    enriched["openAccessPdf"] = {"url": pdf_url}
    enriched["fulltext_status"] = status
    sources = list(enriched.get("sources") or [])
    if "arxiv" not in sources:
        sources.append("arxiv")
    enriched["sources"] = sources
    return enriched


def semantic_fulltext_queries(paper: dict) -> list[str]:
    return fulltext_lookup_queries(paper)


def arxiv_fulltext_queries(paper: dict) -> list[str]:
    title = paper.get("title") or ""
    clean_title = re.sub(r"[:：,，;；.!?()\[\]{}\"']", " ", title)
    clean_title = re.sub(r"\s+", " ", clean_title).strip()
    keywords = title_keywords(title)
    first_author = first_author_name(paper)
    queries = [
        title,
        clean_title,
        " ".join(keywords[:12]),
        " ".join(core_title_keywords(title)),
    ]
    if first_author and keywords:
        queries.append(" ".join(keywords[:8] + [first_author]))
    return dedupe_nonempty_strings(queries)


def dedupe_nonempty_strings(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def title_keywords(title: str) -> list[str]:
    stopwords = {
        "a", "an", "the", "and", "or", "for", "of", "in", "on", "to", "with", "by", "from",
        "via", "using", "towards", "toward", "based", "study", "paper",
    }
    words = re.findall(r"[A-Za-z0-9]+", title or "")
    return [word for word in words if len(word) > 2 and word.lower() not in stopwords]


def core_title_keywords(title: str) -> list[str]:
    keywords = title_keywords(title)
    preferred = [
        word for word in keywords
        if word.lower() in {
            "generative", "generation", "detection", "detecting", "watermark", "watermarking",
            "text", "image", "diffusion", "language", "model", "models", "llm", "pan", "ai",
        }
        or re.fullmatch(r"20\d{2}", word)
    ]
    return preferred[:8] if preferred else keywords[:8]


def first_author_name(paper: dict) -> str:
    authors = paper.get("authors") or []
    if not authors:
        return ""
    first = authors[0]
    if isinstance(first, dict):
        name = first.get("name") or ""
    else:
        name = str(first)
    return name.split()[-1] if name else ""


def is_same_paper_for_fulltext(source_paper: dict, candidate: dict) -> bool:
    return fulltext_candidate_match_reason(source_paper, candidate)[0]


def fulltext_candidate_match_reason(source_paper: dict, candidate: dict) -> tuple[bool, str]:
    source_title = source_paper.get("title") or ""
    candidate_title = candidate.get("title") or ""
    if not source_title or not candidate_title:
        return False, "标题缺失"
    source_key = re.sub(r"\s+", " ", source_title.lower()).strip()
    candidate_key = re.sub(r"\s+", " ", candidate_title.lower()).strip()
    if source_key == candidate_key:
        return True, "标题完全一致"

    similarity = SequenceMatcher(None, source_key, candidate_key).ratio()
    if similarity >= 0.82:
        return True, f"标题相似度 {similarity:.2f}"

    source_keywords = {word.lower() for word in title_keywords(source_title)}
    candidate_keywords = {word.lower() for word in title_keywords(candidate_title)}
    overlap = source_keywords & candidate_keywords
    overlap_ratio = len(overlap) / max(1, min(len(source_keywords), len(candidate_keywords)))
    years_close = bool(source_paper.get("year") and candidate.get("year") and abs(source_paper["year"] - candidate["year"]) <= 1)
    author_overlap = bool(author_names(source_paper) & author_names(candidate))

    if len(overlap) >= 4 and (overlap_ratio >= 0.45 or years_close or author_overlap):
        details = [f"关键词命中 {len(overlap)} 个"]
        if years_close:
            details.append("年份接近")
        if author_overlap:
            details.append("作者部分重合")
        return True, "，".join(details)

    return False, f"未匹配：标题相似度 {similarity:.2f}，关键词命中 {len(overlap)} 个"


def author_names(paper: dict) -> set[str]:
    names = set()
    for author in paper.get("authors") or []:
        if isinstance(author, dict):
            name = author.get("name") or ""
        else:
            name = str(author)
        if name:
            names.add(name.lower())
            names.add(name.split()[-1].lower())
    return names


def extract_arxiv_id_from_paper(paper: dict) -> str:
    values = [paper.get("arxiv_id"), paper.get("paperId"), paper.get("url"), paper.get("doi"), paper_pdf_url(paper)]
    for value in values:
        arxiv_id = extract_arxiv_id_from_text(value)
        if arxiv_id:
            return arxiv_id
    return ""


def update_cached_paper_with_fulltext(paper_key: str, enriched_paper: dict) -> bool:
    result = st.session_state.get("discovery_result") or {}
    papers = result.get("papers") or []
    updated = False
    for index, item in enumerate(papers):
        if paper_state_key(item) == paper_key:
            papers[index] = enriched_paper
            updated = True
            break
    result["papers"] = papers
    st.session_state.discovery_result = result
    st.session_state.last_results = result
    return updated


def auto_enrich_pdfs_for_result(result: dict, limit: int = 10) -> dict:
    enriched_result = dict(result)
    papers = list(enriched_result.get("papers") or [])
    debug_rows = []

    for index, paper in enumerate(papers[:limit]):
        key = paper_state_key(paper)
        try:
            lookup = ensure_fulltext_for_paper(paper)
            enriched_paper = lookup.get("paper") or paper
            lookup_debug = lookup.get("debug") or {}
            lookup_debug["session_state_updated"] = True
            enriched_paper["fulltext_debug"] = lookup_debug
            papers[index] = enriched_paper
            st.session_state.fulltext_debug[key] = lookup_debug
            debug_rows.append(
                {
                    "index": index + 1,
                    "title": paper.get("title") or "API 未提供标题",
                    "status": enriched_paper.get("fulltext_status") or lookup.get("status"),
                    "pdf_url": paper_pdf_url(enriched_paper),
                    "error": lookup.get("error") or "",
                }
            )
        except Exception as exc:
            failed_paper = dict(paper)
            failed_paper["fulltext_status"] = "error"
            failed_paper["fulltext_debug"] = {"executed": True, "failure_step": str(exc), "session_state_updated": True}
            papers[index] = failed_paper
            st.session_state.fulltext_debug[key] = failed_paper["fulltext_debug"]
            debug_rows.append(
                {
                    "index": index + 1,
                    "title": paper.get("title") or "API 未提供标题",
                    "status": "error",
                    "pdf_url": "",
                    "error": str(exc),
                }
            )

    enriched_result["papers"] = papers
    enriched_result["pdf_enrichment_debug"] = debug_rows
    return enriched_result


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    class UploadedPdfBytes:
        def read(self):
            return pdf_bytes

    return extract_text_from_pdf(UploadedPdfBytes())


def show_api_error(exc: Exception) -> None:
    if isinstance(exc, ValueError):
        st.error(str(exc))
    elif isinstance(exc, APIConnectionError):
        st.error("连接 DeepSeek API 失败。请检查网络连接或代理设置后重试。")
    elif isinstance(exc, APIStatusError):
        if exc.status_code == 401:
            st.error("DeepSeek API Key 无效或未授权。请检查 .env 中的 DEEPSEEK_API_KEY。")
        elif exc.status_code == 402:
            st.error("DeepSeek 账户余额不足，请充值后重试。")
        elif exc.status_code == 429:
            st.error("DeepSeek API 请求过于频繁或额度受限，请稍后重试。")
        else:
            st.error(f"DeepSeek API 返回错误：HTTP {exc.status_code}。请检查模型名称、API Key 或账户状态。")
    elif isinstance(exc, APIError):
        st.error(f"DeepSeek API 调用失败：{exc}")
    else:
        st.error(f"生成文献笔记时发生未知错误：{exc}")


def split_markdown_sections(notes: str) -> list[tuple[str, str]]:
    sections = []
    current_title = "文献笔记"
    current_lines = []

    for line in notes.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.replace("## ", "", 1).strip()
            current_lines = []
        elif line.startswith("# "):
            current_title = line.replace("# ", "", 1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return sections or [("文献笔记", notes)]


def render_notes(notes: str) -> None:
    st.subheader("结构化文献笔记")

    for title, body in split_markdown_sections(notes):
        with st.container(border=True):
            st.markdown(f"#### {title}")
            if body:
                st.markdown(body)

    with st.expander("复制/下载 Markdown 文献笔记", expanded=False):
        st.text_area("Markdown 内容", notes, height=280)
        st.download_button(
            "下载 Markdown 文献笔记",
            data=notes.encode("utf-8"),
            file_name="paper_notes.md",
            mime="text/markdown",
        )


def current_source() -> str:
    return st.session_state.get("current_source") or "尚未检索"


SEARCH_MODE_SOURCES = {
    "推荐组合": ["arxiv", "semantic_scholar", "dblp"],
    "最新论文": ["arxiv"],
    "正式发表优先": ["dblp", "semantic_scholar"],
    "引用与摘要优先": ["semantic_scholar"],
}


DATA_SOURCE_DESCRIPTIONS = {
    "arxiv": ("arXiv", "最新预印本，适合追踪新论文"),
    "semantic_scholar": ("Semantic Scholar", "摘要、引用数、开放 PDF 补充"),
    "dblp": ("DBLP", "计算机领域正式发表与 venue 核查"),
}


def data_source_label(source: str) -> str:
    labels = {
        "arxiv": "arXiv",
        "semantic_scholar": "Semantic Scholar",
        "dblp": "DBLP",
        "arXiv": "arXiv",
        "Semantic Scholar": "Semantic Scholar",
        "DBLP": "DBLP",
        "mock": "mock",
    }
    return labels.get(source, source)


def search_scope_text() -> str:
    mode = st.session_state.get("search_mode", "推荐组合")
    selected_sources = st.session_state.get("selected_sources") or SEARCH_MODE_SOURCES["推荐组合"]
    if mode == "自定义":
        return " + ".join(data_source_label(source) for source in selected_sources) if selected_sources else "未选择数据源"
    return mode


def top_search_mode_text() -> str:
    result = st.session_state.get("discovery_result")
    if not result:
        return "待检索"
    if result.get("source") == "mock fallback":
        return "demo fallback"

    selected_sources = result.get("selected_sources") or st.session_state.get("last_selected_sources") or []
    selected_sources = list(selected_sources)
    if len(selected_sources) > 1:
        return "多源联合检索"
    if selected_sources == ["arxiv"]:
        return "arXiv 检索"
    if selected_sources == ["dblp"]:
        return "DBLP 检索"
    if selected_sources == ["semantic_scholar"]:
        return "Semantic Scholar 检索"
    return "待检索"


def sources_for_search_mode(mode: str) -> list[str]:
    if mode == "自定义":
        return list(st.session_state.get("selected_sources") or SEARCH_MODE_SOURCES["推荐组合"])
    return SEARCH_MODE_SOURCES.get(mode, SEARCH_MODE_SOURCES["推荐组合"])


def source_to_mode(source: str) -> str:
    if source == "Semantic Scholar":
        return "真实检索"
    if source == "arXiv fallback":
        return "公开数据源检索"
    if source == "mock fallback":
        return "demo fallback"
    if "+" in source or "DBLP" in source or "arXiv" in source:
        return "多源检索"
    return "尚未检索"


def sync_discovery_status() -> None:
    result = st.session_state.get("discovery_result")
    if result:
        source = result.get("source") or "Semantic Scholar"
    else:
        source = "尚未检索"
    st.session_state.current_source = source
    st.session_state.current_mode = source_to_mode(source)


def current_mode() -> str:
    return st.session_state.get("current_mode") or source_to_mode(current_source())


def render_top_status_bar() -> None:
    deepseek_status = "可用" if os.getenv("DEEPSEEK_API_KEY") else "未配置"

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("DeepSeek 状态", deepseek_status)
        col2.metric("检索范围", search_scope_text())
        col3.metric("检索模式", top_search_mode_text())


def display_source_name(source: str) -> str:
    if source == "arXiv fallback":
        return "arXiv 预印本"
    if source == "mock fallback":
        return "mock 示例数据"
    if source:
        return " / ".join(data_source_label(part.strip()) for part in source.split(" + "))
    return source or "尚未检索"


def render_source_notice(source: str) -> None:
    selected_sources = st.session_state.get("selected_sources") or []
    if selected_sources == ["arxiv"]:
        st.info("当前结果主要用于追踪最新预印本，正式发表信息需要后续核查。")
    elif selected_sources == ["dblp"]:
        st.info("当前结果主要来自计算机领域书目数据，可能缺少摘要、PDF 和引用数。")
    elif len(selected_sources) > 1:
        st.info("系统将合并多源结果，并尽量补充 venue、引用数、PDF 和正式发表状态。")
    elif source == "arXiv fallback":
        st.info("数据源：arXiv 预印本。当前结果适合追踪最新论文，正式发表信息需后续核查。")
    elif source == "mock fallback":
        st.info("当前为 demo 示例数据，可点击重新检索真实论文。")
    elif source == "Semantic Scholar":
        st.caption("数据源：Semantic Scholar。论文元数据来自真实论文 API，适合做基础文献核查。")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("项目说明")
        st.write("面向研究生的科研论文发现与阅读助手。支持从研究方向发现可信论文，也支持上传 PDF 生成结构化文献笔记。")

        st.divider()
        st.header("当前状态")
        if os.getenv("DEEPSEEK_API_KEY"):
            st.success("DeepSeek：可用")
        else:
            st.warning("DeepSeek：未配置")

        if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
            st.success("Semantic Scholar：已配置 API Key")
        else:
            st.caption("论文检索：未配置 Key，将优先尝试公开检索，必要时切换 arXiv。")

        st.write(f"检索范围：{search_scope_text()}")
        st.write(f"检索模式：{top_search_mode_text()}")

        st.divider()
        st.header("使用步骤")
        st.markdown(
            "1. 先输入研究方向，发现并核查论文。\n"
            "2. 查看推荐理由与核查状态，筛选值得精读的文献。\n"
            "3. 在阅读 PDF 中上传论文，生成结构化文献笔记。"
        )


def render_pdf_notes_tab() -> None:
    uploaded_file = st.file_uploader("上传 PDF 论文", type=["pdf"])
    use_demo_fallback = st.checkbox(
        "使用示例输出进行演示",
        help="课堂演示时 API 不可用也能展示完整效果。",
    )

    if uploaded_file is None:
        st.info("请先上传 PDF 文件。")
        return

    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.active_file_id != file_id:
        st.session_state.active_file_id = file_id
        st.session_state.notes = ""
        st.session_state.extracted_text = ""
        st.session_state.cleaned_text = ""
        st.session_state.pdf_processed = False

    try:
        if not st.session_state.pdf_processed:
            read_status = st.empty()
            read_progress = st.progress(0, text="准备读取 PDF...")

            extracted_text = extract_text_from_pdf(
                uploaded_file,
                progress_bar=read_progress,
                status_text=read_status,
            )

            read_status.info("正在清洗 PDF 文本...")
            cleaned_text = clean_paper_text(extracted_text)
            st.session_state.extracted_text = extracted_text
            st.session_state.cleaned_text = cleaned_text
            st.session_state.pdf_processed = True

            read_progress.progress(100, text="PDF 读取与文本清洗完成")
            read_status.success("PDF 读取完成，文本已完成基础清洗。")
        else:
            extracted_text = st.session_state.extracted_text
            cleaned_text = st.session_state.cleaned_text

        if not extracted_text:
            st.warning("没有提取到文本。该 PDF 可能是扫描版图片，需要 OCR 才能解析。")
            return

        st.success("PDF 文本提取和清洗完成，可以生成结构化文献笔记。")

        if st.button("生成结构化文献笔记", type="primary"):
            if use_demo_fallback:
                st.session_state.notes = SAMPLE_MARKDOWN_NOTES
                st.info("当前使用示例输出进行演示，没有调用 DeepSeek API。")
            else:
                try:
                    generate_status = st.empty()
                    generate_progress = st.progress(0, text="准备生成结构化文献笔记...")

                    generate_status.info("正在准备模型输入...")
                    generate_progress.progress(25, text="已截取清洗后前 15000 个字符")

                    generate_status.info("DeepSeek 正在生成结构化文献笔记，请稍候...")
                    generate_progress.progress(65, text="DeepSeek 正在生成...")
                    st.session_state.notes = generate_structured_notes(cleaned_text)

                    generate_progress.progress(100, text="结构化文献笔记生成完成")
                    generate_status.success("结构化文献笔记已生成。")
                except Exception as exc:
                    show_api_error(exc)

        if st.session_state.notes:
            render_notes(st.session_state.notes)

        with st.expander("调试信息：原始文本与清洗文本", expanded=False):
            col1, col2 = st.columns(2)
            col1.metric("原始提取文本长度", len(extracted_text))
            col2.metric("清洗后文本长度", len(cleaned_text))
            st.write(f"AI 输入长度上限：前 {MAX_AI_INPUT_CHARS} 个字符")
            st.text_area("清洗后前 2000 个字符", cleaned_text[:2000], height=260)
            st.text_area("原始提取文本前 2000 个字符", extracted_text[:2000], height=260)
    except Exception as exc:
        st.error("PDF 解析失败，请检查文件是否为有效 PDF。")
        st.exception(exc)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_discover_papers(topic: str, deepseek_model: str, has_semantic_key: bool, selected_sources: tuple[str, ...], search_mode: str) -> dict:
    return discover_papers(topic, limit=10, selected_sources=list(selected_sources))


def render_discovery_tab() -> None:
    st.subheader("发现论文")
    st.caption("DeepSeek 只用于扩展英文关键词和推荐理由；论文标题、作者、年份、venue、URL 和 PDF 链接必须来自真实数据源。")

    topic = st.text_input("研究方向或关键词", value="", placeholder="例如：AIGC模型水印")
    search_mode = st.pills(
        "检索模式",
        options=["推荐组合", "最新论文", "正式发表优先", "引用与摘要优先", "自定义"],
        default=st.session_state.get("search_mode", "推荐组合"),
        key="search_mode",
        help="先选择适合当前任务的检索模式；需要精细控制时再使用自定义。",
    )
    search_mode = search_mode or "推荐组合"

    if search_mode == "自定义":
        st.caption("自定义数据源")
        selected_sources = []
        source_cols = st.columns(3)
        for source, column in zip(["arxiv", "semantic_scholar", "dblp"], source_cols):
            label, description = DATA_SOURCE_DESCRIPTIONS[source]
            toggle_key = f"custom_source_{source}"
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = source in (st.session_state.get("selected_sources") or SEARCH_MODE_SOURCES["推荐组合"])
            with column:
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.caption(description)
                    selected = st.toggle("已选择", key=toggle_key)
                    if selected:
                        st.success("已选择")
                        selected_sources.append(source)
                    else:
                        st.caption("未选择")
    else:
        selected_sources = sources_for_search_mode(search_mode)
        st.caption(f"当前检索范围：{' + '.join(data_source_label(source) for source in selected_sources)}")

    selected_sources_tuple = tuple(selected_sources)
    st.session_state.selected_sources = list(selected_sources_tuple)

    search_col, refresh_col = st.columns([1.4, 1])
    search_requested = search_col.button("扩展关键词并检索论文", type="primary")
    refresh_requested = refresh_col.button("重新检索真实论文")

    if refresh_requested:
        st.session_state.force_refresh = True
        if (st.session_state.get("discovery_result") or {}).get("source") == "mock fallback":
            st.session_state.discovery_result = None
            st.session_state.last_results = None
            st.session_state.last_search_status = "failed"
            st.session_state.last_source = ""
        st.session_state.fulltext_lookup_cache = {}
        st.session_state.fulltext_debug = {}
        cached_discover_papers.clear()

    if search_requested or refresh_requested:
        normalized_topic = topic.strip()
        if not normalized_topic:
            st.warning("请先输入研究方向或关键词。")
            st.session_state.force_refresh = False
            return
        if not selected_sources_tuple:
            st.warning("请至少选择一个论文数据源。")
            st.session_state.force_refresh = False
            return

        force_refresh = bool(st.session_state.get("force_refresh"))
        last_query = st.session_state.get("last_query") or st.session_state.get("discovery_query")
        last_result = st.session_state.get("last_results") or st.session_state.get("discovery_result")
        last_status = st.session_state.get("last_search_status") or determine_search_status(last_result)
        last_source = st.session_state.get("last_source") or (last_result or {}).get("source", "")
        last_selected_sources = tuple(st.session_state.get("last_selected_sources") or ())
        last_search_mode = st.session_state.get("last_search_mode", "")
        same_source_selection = last_selected_sources == selected_sources_tuple
        same_search_mode = last_search_mode == search_mode

        if same_source_selection and same_search_mode and can_reuse_discovery_result(
            current_query=normalized_topic,
            last_query=last_query,
            last_result=last_result,
            last_search_status=last_status,
            last_source=last_source,
            force_refresh=force_refresh,
        ):
            st.session_state.discovery_query = normalized_topic
            st.session_state.discovery_result = last_result
            st.session_state.selected_sources = list(selected_sources_tuple)
            sync_discovery_status()
            st.info("已复用上一次真实检索结果。")
            st.session_state.force_refresh = False
        else:
            try:
                if force_refresh or last_status != "success" or last_source == "mock fallback":
                    cached_discover_papers.clear()

                status = st.empty()
                progress = st.progress(0, text="准备检索式...")

                status.info("正在准备英文检索式；如果 DeepSeek 响应较慢，将自动使用本地 fallback。")
                progress.progress(30, text="正在准备检索式")

                result = cached_discover_papers(
                    normalized_topic,
                    os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
                    bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
                    selected_sources_tuple,
                    search_mode,
                )

                progress.progress(80, text="正在合并、去重、评分和核查论文结果")
                status.info("正在自动查找开放 PDF...")
                result = auto_enrich_pdfs_for_result(result, limit=10)
                search_status = determine_search_status(result)
                source = result.get("source") or ""
                st.session_state.discovery_query = normalized_topic
                st.session_state.discovery_result = result
                st.session_state.last_query = normalized_topic
                st.session_state.last_results = result
                st.session_state.last_search_status = search_status
                st.session_state.last_source = source
                st.session_state.last_selected_sources = selected_sources_tuple
                st.session_state.last_search_mode = search_mode
                st.session_state.selected_sources = list(selected_sources_tuple)
                sync_discovery_status()

                if search_status != "success" or source == "mock fallback":
                    cached_discover_papers.clear()

                progress.progress(100, text="论文发现与核查完成")
                status.success("已完成论文检索与核查。")
                st.session_state.force_refresh = False
                st.rerun()
            except Exception as exc:
                st.session_state.last_query = normalized_topic
                st.session_state.last_results = None
                st.session_state.last_search_status = "failed"
                st.session_state.last_source = ""
                st.session_state.force_refresh = False
                cached_discover_papers.clear()
                st.error(f"论文发现流程失败：{exc}")

    result = st.session_state.get("discovery_result")
    if not result:
        return

    source = result.get("source") or "Semantic Scholar"
    render_source_notice(source)

    papers = result.get("papers") or []
    if not papers:
        render_discovery_debug(result)
        st.info("没有检索到通过核查的论文。可以尝试更换关键词。")
        return

    render_discovery_summary(result, papers)
    render_sorting_rules()
    render_discovery_debug(result)
    render_top_recommendations(papers)
    st.info("下一步操作：打开感兴趣论文的 PDF 后，可切换到‘阅读 PDF’页面生成结构化文献笔记。")

    st.markdown("#### 完整论文列表（前 10 篇）")
    for index, paper in enumerate(papers[:10], start=1):
        render_paper_card(index, paper)


def render_library_tab() -> None:
    st.subheader("我的文献库")
    st.caption("当前为本地 session 版本，不需要登录，也不会写入数据库。")

    if not st.session_state.library_papers:
        st.info("还没有收藏论文。可以在“发现论文”结果卡片中点击“加入待读”或标记阅读状态。")
        return

    st.download_button(
        "导出我的文献库为 Markdown",
        data=library_markdown().encode("utf-8"),
        file_name="my_paper_library.md",
        mime="text/markdown",
    )

    grouped = group_library_by_status()
    for status in ["待读", "精读", "已读", "正在读", "不相关", "未读"]:
        papers = grouped.get(status, [])
        if not papers:
            continue
        with st.expander(f"{status}（{len(papers)}）", expanded=status in {"待读", "精读"}):
            for paper in papers:
                with st.container(border=True):
                    st.markdown(f"#### {paper.get('title') or 'API 未提供标题'}")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.write(f"**年份：** {paper.get('year') or 'API 未提供'}")
                    col2.write(f"**Venue：** {paper.get('venue') or 'API 未提供'}")
                    col3.write(f"**来源：** {paper.get('source') or 'API 未提供'}")
                    col4.write(f"**全文笔记：** {'已有笔记' if paper.get('has_full_notes') else '暂无笔记'}")

                    status_options = ["未读", "待读", "正在读", "已读", "精读", "不相关"]
                    current_status = paper.get("status") or "未读"
                    new_status = st.selectbox(
                        "阅读状态",
                        status_options,
                        index=status_options.index(current_status) if current_status in status_options else 0,
                        key=f"library_status_{paper['key']}",
                    )
                    if new_status != current_status:
                        st.session_state.library_papers[paper["key"]]["status"] = new_status
                        st.rerun()

                    link_cols = st.columns([1, 1, 4])
                    if paper.get("url"):
                        link_cols[0].link_button("论文链接", paper["url"], key=f"library_url_{paper['key']}")
                    else:
                        link_cols[0].button("论文链接", disabled=True, key=f"library_url_disabled_{paper['key']}")
                    if paper.get("pdf_url"):
                        link_cols[1].link_button("PDF", paper["pdf_url"], key=f"library_pdf_{paper['key']}")
                    else:
                        link_cols[1].button("PDF", disabled=True, key=f"library_pdf_disabled_{paper['key']}")


def render_discovery_summary(result: dict, papers: list[dict]) -> None:
    source = result.get("source") or "Semantic Scholar"
    count = len(papers)
    recent_count = sum(1 for paper in papers if (paper.get("year") or 0) >= 2024)
    pdf_count = sum(1 for paper in papers if paper_pdf_url(paper))
    high_priority_count = sum(1 for paper in papers if paper.get("recommendation_priority") == "高")
    selected_sources = result.get("selected_sources") or []
    hit_sources = sorted({source for paper in papers for source in paper.get("sources", []) if source})

    if source == "Semantic Scholar":
        source_text = "Semantic Scholar"
        usage_text = "这些结果适合作为论文检索和元数据核查的起点，可优先筛选高相关、有 PDF、引用较高的论文继续精读。"
    elif source == "arXiv fallback":
        source_text = "arXiv 预印本"
        usage_text = "这些结果适合追踪最新预印本，正式发表 venue 需要后续通过 Semantic Scholar、DBLP 或 Crossref 进一步核查。"
    elif source == "mock fallback":
        source_text = "mock 示例数据"
        usage_text = "当前结果仅用于课堂 demo 验证页面流程，不应作为真实文献推荐。"
    elif "DBLP" in source and "arXiv" not in source and "Semantic Scholar" not in source:
        source_text = "DBLP"
        usage_text = "这些结果主要来自计算机领域书目数据，可能缺少摘要、PDF 和引用数。"
    else:
        source_text = display_source_name(source)
        usage_text = "系统已合并多源结果，并尽量补充 venue、引用数、PDF 和正式发表状态。"

    with st.container(border=True):
        st.markdown("#### 助手总结")
        st.write(
            f"本次找到 {count} 篇可展示论文，当前数据源为 **{source_text}**。{usage_text}"
        )
        st.write(f"**本次检索范围：** {' + '.join(data_source_label(source) for source in selected_sources) if selected_sources else '未提供'}")
        st.write(f"**本次实际命中来源：** {' + '.join(data_source_label(source) for source in hit_sources) if hit_sources else '未提供'}")
        if len(selected_sources) > 1:
            st.caption("系统已对多源结果进行合并与去重。")
        st.markdown("**阅读建议**")
        advice_items = [
            f"优先查看推荐优先级为“高”的论文（当前 {high_priority_count} 篇），再筛选“中”等级论文。",
            f"优先打开带 PDF 的论文（当前 {pdf_count} 篇），便于快速核查方法、实验和结论。",
        ]
        if source == "arXiv fallback":
            advice_items.append("arXiv 结果适合追踪最新预印本，正式发表 venue 需要后续核查。")
        elif recent_count:
            advice_items.append(f"可先关注近两年的论文（当前 {recent_count} 篇），再补充高引用背景文献。")
        else:
            advice_items.append("当前结果较适合作为背景线索，建议结合关键词继续扩大检索。")
        for item in advice_items:
            st.markdown(f"- {item}")


def render_discovery_debug(result: dict) -> None:
    with st.expander("检索诊断信息", expanded=False):
        debug = result.get("debug_info") or {}
        source_debug = debug.get("sources") or {}
        st.caption(f"技术数据源：{result.get('source') or 'unknown'}")
        st.write(f"**用户原始 query：** {debug.get('user_query') or result.get('topic') or '未提供'}")
        st.write(f"**实际 selected_sources：** {' / '.join(data_source_label(source) for source in (result.get('selected_sources') or [])) or '未提供'}")
        st.markdown("**英文关键词**")
        st.write(", ".join(result.get("keywords") or ["API 未提供"]))
        st.caption(f"关键词来源：{result.get('keyword_expansion_source') or 'DeepSeek'}")

        arxiv_debug = source_debug.get("arxiv", {})
        semantic_debug = source_debug.get("semantic_scholar", {})
        dblp_debug = source_debug.get("dblp", {})
        diag_cols = st.columns(3)
        diag_cols[0].write(f"**arXiv 是否被调用：** {'是' if arxiv_debug.get('called') else '否'}")
        diag_cols[0].write(f"**arXiv 返回条数：** {arxiv_debug.get('count', 0)}")
        diag_cols[0].write(f"**arXiv 失败原因：** {arxiv_debug.get('error') or '无'}")
        diag_cols[1].write(f"**Semantic Scholar 是否被调用：** {'是' if semantic_debug.get('called') else '否'}")
        diag_cols[1].write(f"**Semantic Scholar 返回条数：** {semantic_debug.get('count', 0)}")
        diag_cols[1].write(f"**Semantic Scholar 失败原因：** {semantic_debug.get('error') or '无'}")
        diag_cols[2].write(f"**DBLP 是否被调用：** {'是' if dblp_debug.get('called') else '否'}")
        diag_cols[2].write(f"**DBLP 返回条数：** {dblp_debug.get('count', 0)}")
        diag_cols[2].write(f"**DBLP 失败原因：** {dblp_debug.get('error') or '无'}")
        st.write(f"**合并前总论文数：** {debug.get('pre_merge_count', 0)}")
        st.write(f"**去重后论文数：** {debug.get('deduplicated_count', 0)}")
        st.write(f"**最终为什么进入 mock fallback：** {debug.get('mock_fallback_reason') or '未进入 mock fallback'}")
        if result.get("pdf_enrichment_debug"):
            st.markdown("**PDF 自动补全结果**")
            st.dataframe(result.get("pdf_enrichment_debug"), use_container_width=True)

        st.markdown("**Semantic Scholar 检索式**")
        for query in result.get("search_queries") or [result.get("search_query")]:
            st.code(query or "", language="text")

        detailed_warning = result.get("warning")
        product_notice = result.get("source_warning")
        if result.get("keyword_expansion_warning"):
            st.markdown("**关键词扩展提示**")
            st.info(result["keyword_expansion_warning"])
        if detailed_warning and detailed_warning != product_notice:
            st.markdown("**API warning / fallback 详情**")
            st.info(detailed_warning)
        elif product_notice and result.get("source") != "arXiv fallback":
            st.markdown("**API warning / fallback 详情**")
            st.info(product_notice)
        if result.get("debug_info"):
            st.markdown("**完整 debug_info**")
            st.json(result.get("debug_info"))
        st.caption(f"Semantic Scholar 认证模式：{result.get('semantic_scholar_auth_mode') or 'unknown'}")


def top_recommendation_score(paper: dict) -> tuple:
    priority_rank = {"高": 3, "中": 2, "低": 1}.get(display_priority(paper), 1)
    return (
        priority_rank,
        1 if title_is_strongly_related(paper) else 0,
        1 if paper_pdf_url(paper) else 0,
        paper.get("year") or 0,
        paper.get("score") or 0,
    )


def select_top_recommendations(papers: list[dict], limit: int = 3) -> list[dict]:
    return sorted(papers, key=top_recommendation_score, reverse=True)[:limit]


def render_top_recommendations(papers: list[dict]) -> None:
    top_papers = select_top_recommendations(papers, limit=3)
    if not top_papers:
        return

    st.markdown("#### 优先关注论文")
    columns = st.columns(3)
    for index, paper in enumerate(top_papers, start=1):
        with columns[index - 1]:
            with st.container(border=True):
                st.caption(f"Top {index}")
                st.markdown(f"**{paper.get('title') or 'API 未提供标题'}**")
                st.write(f"**推荐原因：** {display_short_recommendation_reason(paper)}")
                st.write(f"**阅读定位：** {reading_position(paper)}")


def render_sorting_rules() -> None:
    with st.expander("排序依据说明", expanded=False):
        st.markdown(
            "- 关键词相关性：标题或摘要命中扩展关键词会加分。\n"
            "- 近三年优先：较新的论文会获得时间加权。\n"
            "- 有 PDF 优先：存在 open access PDF 链接会加分。\n"
            "- 引用数较高优先：引用数越高，分数适度提高。\n"
            "- 顶会/顶刊 venue 加权：只对可识别的优质 venue 加分；arXiv 不按顶会/顶刊处理。"
        )


def paper_display_source(paper: dict) -> str:
    sources = paper.get("sources") or []
    if sources:
        return " / ".join(data_source_label(source) for source in sources)
    source = paper.get("_source") or ""
    venue = (paper.get("venue") or "").lower()
    if source == "arXiv fallback" or venue == "arxiv":
        return "arXiv 预印本"
    if source == "Mock fallback":
        return "mock 示例数据"
    return "Semantic Scholar"


def source_check_text(paper: dict) -> str:
    source = paper.get("_source") or ""
    if source == "Mock fallback":
        return "待确认"
    if source in {"Semantic Scholar", "semantic_scholar", "arXiv fallback", "arxiv", "dblp", "DBLP"} or paper.get("url") or paper.get("paperId"):
        return "可追溯"
    return "待确认"


def metadata_text(paper: dict, verification: dict) -> str:
    value = paper.get("metadata_verified") or verification.get("metadata_verified")
    if value == "完整":
        return "完整"
    return "部分完整"


def quality_text(paper: dict, verification: dict) -> str:
    value = paper.get("quality_verified") or verification.get("quality_verified") or "待进一步确认"
    source = paper.get("_source") or ""
    venue = paper.get("venue") or ""
    if paper.get("dblp_status") in {"found", "已找到"}:
        return "正式发表，venue 已确认" if paper.get("dblp_venue") else "正式发表"
    if source == "Mock fallback":
        return "待进一步确认"
    if source == "arXiv fallback" or venue.lower() == "arxiv":
        return "arXiv 预印本"
    if value == "顶会/顶刊":
        return "顶会/顶刊"
    if venue:
        return "正式发表"
    return "待进一步确认"


def dblp_status_text(paper: dict) -> str:
    if paper.get("_source") == "Mock fallback":
        return "示例数据不核查"
    status = paper.get("dblp_status") or "not_checked"
    mapping = {
        "found": "已找到",
        "not_found": "未找到",
        "error": "未完成",
        "not_checked": "未执行",
        "已找到": "已找到",
        "未找到": "未找到",
        "未执行": "未执行",
    }
    return mapping.get(status, "未执行")


def dblp_venue_text(paper: dict) -> str:
    return paper.get("dblp_venue") or "未找到"


def formal_publication_text(paper: dict) -> str:
    if paper.get("publication_status") == "confirmed":
        return "已确认"
    if paper.get("publication_status") == "pending":
        return "待确认"
    return paper.get("formal_publication_status") or "待确认"


def publication_status_note(paper: dict) -> str:
    if paper.get("_source") == "arXiv fallback" and paper.get("dblp_status") in {"found", "已找到"}:
        return "已在 DBLP 找到正式发表记录，venue 信息已补充。"
    if paper.get("_source") == "arXiv fallback":
        return "arXiv 预印本，DBLP 暂未找到正式发表记录。"
    if paper.get("dblp_status") == "error":
        return "DBLP 核查未完成。"
    return ""


def compact_verification_tags(paper: dict, source_status: str, metadata_status: str, quality_status: str) -> list[str]:
    source_tag = "来源可追溯" if source_status == "可追溯" else "来源待确认"
    metadata_tag = "元数据完整" if metadata_status == "完整" else "元数据部分完整"

    if quality_status == "arXiv 预印本" or paper.get("_source") == "arXiv fallback":
        quality_tag = "arXiv 预印本"
    elif paper.get("formal_publication_status") == "已确认":
        quality_tag = "正式发表已确认"
    elif "正式发表" in quality_status:
        quality_tag = "正式发表待确认"
    else:
        quality_tag = "正式发表待确认"

    return [source_tag, metadata_tag, quality_tag]


def compact_dblp_tags(paper: dict) -> list[str]:
    tags = [f"DBLP 核查：{dblp_status_text(paper)}", f"正式发表状态：{formal_publication_text(paper)}"]
    venue = paper.get("dblp_venue")
    if venue:
        tags.append(f"DBLP venue：{venue}")
    return tags


def render_compact_tags(tags: list[str]) -> None:
    tag_html = "".join(
        (
            '<span style="display:inline-block;margin:0 0.35rem 0.35rem 0;'
            'padding:0.18rem 0.55rem;border:1px solid rgba(128,128,128,0.28);'
            'border-radius:999px;background:rgba(128,128,128,0.10);font-size:0.82rem;">'
            f"{tag}</span>"
        )
        for tag in tags
    )
    st.markdown(tag_html, unsafe_allow_html=True)


def display_short_recommendation_reason(paper: dict) -> str:
    paper_type = classify_paper_type(paper)
    if paper_type == "综述类":
        return "适合先建立方向背景和研究脉络。"
    if paper_type == "方法类":
        return "适合重点理解方法设计和实验验证。"
    if paper_type == "检测类":
        return "适合关注水印检测、验证和评估方式。"
    if paper_type == "攻击/鲁棒性类":
        return "适合补充攻击场景和鲁棒性风险视角。"
    if paper_type in {"LLM 水印", "图像/扩散模型水印", "多模态水印"}:
        return f"适合追踪{paper_type}这一子方向。"
    if paper.get("_source") == "arXiv fallback":
        return "适合追踪最新预印本线索。"
    return "适合作为相关方向的候选阅读材料。"


def display_recommendation_reason(paper: dict) -> str:
    title = paper.get("title") or "该论文"
    year = paper.get("year") or "年份未提供"
    venue = paper.get("venue") or "venue 未提供"
    hits = paper.get("hit_keywords") or []
    source = paper_display_source(paper)
    paper_type = classify_paper_type(paper)

    parts = [f"《{title}》更接近“{paper_type}”，来自 {source}，年份为 {year}，venue 为 {venue}。"]
    if hits:
        parts.append(f"标题或摘要命中了“{'、'.join(hits[:3])}”等关键词，和当前研究方向存在直接关联。")
    elif title_is_strongly_related(paper):
        parts.append("标题本身包含水印和模型/生成内容相关线索，适合作为优先候选。")
    elif paper.get("abstract"):
        parts.append("摘要提供了可核查的研究内容，可先用来判断是否覆盖你的具体问题。")
    if paper_type == "综述类":
        parts.append("它适合帮助快速建立方向地图，梳理已有方法、问题和研究分支。")
    elif paper_type == "检测类":
        parts.append("它适合用来理解水印检测、验证或识别环节的评估思路。")
    elif paper_type == "攻击/鲁棒性类":
        parts.append("它适合补充水印方案在攻击、移除或鲁棒性场景下的风险视角。")
    elif paper_type in {"LLM 水印", "图像/扩散模型水印", "多模态水印"}:
        parts.append(f"它聚焦 {paper_type}，适合和你的研究方向做子领域对齐。")
    else:
        parts.append("它适合作为方法或相关工作候选，后续需要结合原文进一步判断贡献。")
    return "".join(parts)


def display_reading_advice(paper: dict) -> str:
    paper_type = classify_paper_type(paper)
    source = paper.get("_source") or ""
    venue = paper.get("venue") or ""
    arxiv_note = ""
    if source == "arXiv fallback" or venue.lower() == "arxiv":
        arxiv_note = " 该论文来自 arXiv 预印本，正式发表信息需要后续核查。"

    if paper_type == "综述类":
        return "适合作为方向入门或背景梳理；建议先读 Introduction 和 Related Work，再整理该方向的问题脉络。" + arxiv_note
    if paper_type == "方法类":
        return "适合做方法精读；建议重点看 Method、Experiments 和 Limitations，确认方法假设与实验设置。" + arxiv_note
    if paper_type == "检测类":
        return "适合关注水印检测或验证思路；建议重点看 Threat Model、Evaluation 和 Limitations。" + arxiv_note
    if paper_type == "攻击/鲁棒性类":
        return "适合了解水印方案的失效场景；建议重点看 Threat Model、Attack Setting、Evaluation 和 Limitations。" + arxiv_note
    if paper_type == "LLM 水印":
        return "适合关注文本生成或大语言模型水印；建议重点看 Introduction、Method 和 Experiments。" + arxiv_note
    if paper_type == "图像/扩散模型水印":
        return "适合关注图像生成或扩散模型水印；建议重点看 Method、Experiments 和视觉质量评估。" + arxiv_note
    if paper_type == "多模态水印":
        return "适合了解跨模态水印线索；建议重点看任务定义、Method 和 Evaluation。" + arxiv_note
    return "适合作为相关方向补充阅读；建议先快速浏览 Abstract、Introduction 和 Conclusion。" + arxiv_note


def paper_search_text(paper: dict) -> str:
    return f"{paper.get('title') or ''} {paper.get('abstract') or ''}".lower()


def classify_paper_type(paper: dict) -> str:
    text = paper_search_text(paper)
    title = (paper.get("title") or "").lower()

    if any(word in text for word in ["survey", "review", "taxonomy", "systematic literature"]):
        return "综述类"
    if any(word in text for word in ["attack", "robust", "robustness", "removal", "evasion", "spoof", "jailbreak"]):
        return "攻击/鲁棒性类"
    if any(word in text for word in ["detect", "detection", "detector", "identify", "identification", "verification"]):
        return "检测类"
    if any(word in text for word in ["multimodal", "multi-modal", "cross-modal", "vision-language"]):
        return "多模态水印"
    if any(word in text for word in ["diffusion", "image", "visual", "stable diffusion", "latent diffusion"]):
        return "图像/扩散模型水印"
    if any(word in text for word in ["llm", "large language model", "language model", "text generation", "gpt"]):
        return "LLM 水印"
    if any(word in title for word in ["watermark", "watermarking"]) or any(word in text for word in ["method", "framework", "algorithm", "approach"]):
        return "方法类"
    return "其他相关方向"


def title_is_strongly_related(paper: dict) -> bool:
    title = (paper.get("title") or "").lower()
    hits = [keyword.lower() for keyword in paper.get("hit_keywords") or []]
    if "watermark" in title and any(word in title for word in ["ai", "generative", "llm", "diffusion", "model", "content", "language", "image"]):
        return True
    return any(hit and hit in title for hit in hits)


def display_priority(paper: dict) -> str:
    year = paper.get("year") or 0
    recent = year >= 2024
    has_pdf = bool(paper_pdf_url(paper))
    strong_title = title_is_strongly_related(paper)
    has_metadata = bool(paper.get("title") and (paper.get("abstract") or paper.get("venue")))
    paper_type = classify_paper_type(paper)
    related = strong_title or bool(paper.get("hit_keywords")) or "watermark" in paper_search_text(paper)

    if (strong_title and recent and has_pdf) or (paper_type == "综述类" and strong_title):
        return "高"
    if related and has_metadata:
        return "中"
    return "低"


def citation_display(paper: dict) -> str:
    citation_count = paper.get("citation_count")
    if citation_count is None:
        citation_count = paper.get("citationCount")
    if citation_count is None and "semantic_scholar" not in (paper.get("sources") or []) and "Semantic Scholar" not in (paper.get("sources") or []):
        return "当前数据源未提供，待 Semantic Scholar 补充"
    if citation_count is None:
        return "API 未提供"
    return str(citation_count)


def pdf_status_display(paper: dict) -> str:
    if paper_pdf_url(paper):
        return "有 PDF"
    status = paper.get("fulltext_status") or ""
    if status == "checking":
        return "查找中"
    if status == "not_found":
        return "未找到开放 PDF"
    if status == "error":
        return "查找失败，可手动上传"
    return "未找到开放 PDF"


def reading_position(paper: dict) -> str:
    paper_type = classify_paper_type(paper)
    source = paper.get("_source") or ""
    if paper_type == "综述类":
        return "综述入门"
    if source == "arXiv fallback" and (paper.get("year") or 0) >= 2024:
        return "最新进展"
    if paper_type in {"方法类", "检测类", "攻击/鲁棒性类", "LLM 水印", "图像/扩散模型水印", "多模态水印"}:
        return "方法精读"
    return "扩展阅读"


def render_inline_notes(notes: str, download_key: str, file_name: str) -> None:
    for title, body in split_markdown_sections(notes):
        with st.container(border=True):
            st.markdown(f"#### {title}")
            if body:
                st.markdown(body)

    st.download_button(
        "下载 Markdown 文献笔记",
        data=notes.encode("utf-8"),
        file_name=file_name,
        mime="text/markdown",
        key=download_key,
    )


def inject_reading_dialog_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[role="dialog"] {
            width: min(84vw, 1120px) !important;
            max-width: min(84vw, 1120px) !important;
            max-height: 84vh !important;
        }

        div[data-testid="stDialog"] div[role="dialog"] > div {
            max-height: 84vh !important;
        }

        div[data-testid="stDialog"] h1,
        div[data-testid="stDialog"] h2,
        div[data-testid="stDialog"] h3 {
            white-space: normal !important;
            line-height: 1.28 !important;
        }

        div[data-testid="stDialog"] .stMarkdown {
            line-height: 1.72;
        }

        div[data-testid="stDialog"] .stMarkdown p {
            margin-bottom: 0.75rem;
        }

        div[data-testid="stDialog"] .stMarkdown h2,
        div[data-testid="stDialog"] .stMarkdown h3,
        div[data-testid="stDialog"] .stMarkdown h4 {
            margin-top: 1.15rem;
            margin-bottom: 0.6rem;
        }

        div[data-testid="stDialog"] [data-testid="stVerticalBlock"] {
            gap: 0.75rem;
        }

        div[data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px;
        }

        .reading-dialog-actions {
            border-top: 1px solid rgba(128, 128, 128, 0.25);
            margin-top: 0.75rem;
            padding-top: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_paper_actions(paper: dict, pdf_url: str, paper_url: str) -> None:
    key = paper_state_key(paper)
    topic = st.session_state.get("discovery_query", "")
    title = paper.get("title") or "API 未提供标题"

    st.markdown("**操作**")
    quick_col, full_col, page_col, pdf_col = st.columns([1.2, 1.7, 1, 1])

    if quick_col.button("快速判断", key=f"quick_read_{key}"):
        if key not in st.session_state.quick_readings:
            try:
                with st.spinner("正在基于标题和摘要生成快速解读..."):
                    st.session_state.quick_readings[key] = generate_quick_reading(paper, topic)
            except Exception as exc:
                show_api_error(exc)
                return
        st.session_state.active_reading_dialog = {
            "type": "quick",
            "key": key,
            "title": title,
        }

    if full_col.button("生成全文文献笔记", key=f"full_note_{key}"):
        if key not in st.session_state.full_reading_notes:
            try:
                if pdf_url:
                    st.info("已找到 PDF，正在生成全文文献笔记。")
                    fulltext_result = ensure_fulltext_for_paper(paper)
                else:
                    st.info("当前论文未直接提供 PDF，正在自动查找开放全文。")
                    fulltext_result = ensure_fulltext_for_paper(paper)
                    if fulltext_result["status"] == "fulltext_found_by_enrichment":
                        st.success(fulltext_result["message"])

                if fulltext_result["status"] == "fulltext_not_found":
                    st.session_state.active_reading_dialog = {
                        "type": "full_error",
                        "key": key,
                        "title": title,
                        "error": fulltext_result.get("message") or "未找到开放全文。",
                        "debug": fulltext_result.get("debug") or {},
                    }
                    return

                enriched_paper = fulltext_result.get("paper") or paper
                st.session_state.fulltext_debug[key] = fulltext_result.get("debug") or {}
                session_updated = update_cached_paper_with_fulltext(key, enriched_paper)
                st.session_state.fulltext_debug[key]["session_state_updated"] = session_updated
                resolved_pdf_url = fulltext_result["pdf_url"]
                with st.spinner("正在下载并读取 PDF 全文..."):
                    pdf_bytes = download_pdf_bytes(resolved_pdf_url, timeout=30)
                    extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
                    cleaned_text = clean_paper_text(extracted_text)
                    if not cleaned_text:
                        raise RuntimeError("PDF 文本为空。")
            except Exception as exc:
                st.session_state.active_reading_dialog = {
                    "type": "full_error",
                    "key": key,
                    "title": title,
                    "error": str(exc),
                    "debug": fulltext_result.get("debug") if "fulltext_result" in locals() else {},
                }
                return

            try:
                with st.spinner("正在基于 PDF 全文生成结构化文献笔记..."):
                    st.session_state.full_reading_notes[key] = generate_structured_notes(cleaned_text)
                    update_library_note_status(key)
            except Exception as exc:
                show_api_error(exc)
                return
        st.session_state.active_reading_dialog = {
            "type": "full",
            "key": key,
            "title": title,
        }

    if not pdf_url:
        st.caption("系统已自动尝试查找开放 PDF。未找到开放 PDF，可手动上传 PDF。")

    library_col1, library_col2, library_col3 = st.columns(3)
    if library_col1.button("加入待读", key=f"library_todo_{key}"):
        add_or_update_library_paper(paper, "待读")
        st.success("已加入待读列表。")
    if library_col2.button("标记为精读", key=f"library_deep_{key}"):
        add_or_update_library_paper(paper, "精读")
        st.success("已标记为精读。")
    if library_col3.button("标记为不相关", key=f"library_irrelevant_{key}"):
        add_or_update_library_paper(paper, "不相关")
        st.info("已标记为不相关。")

    if paper_url:
        page_col.link_button("论文页面", paper_url, key=f"paper_page_{key}", use_container_width=True)
    else:
        page_col.button("论文页面", key=f"paper_page_disabled_{key}", disabled=True)

    if pdf_url:
        pdf_col.link_button("PDF", pdf_url, key=f"paper_pdf_{key}", use_container_width=True)
    else:
        pdf_col.button("PDF", key=f"paper_pdf_disabled_{key}", disabled=True)


def close_reading_dialog() -> None:
    st.session_state.active_reading_dialog = None
    st.rerun()


def render_reading_dialog_body(active: dict) -> None:
    result_type = active.get("type")
    key = active.get("key")

    if result_type == "quick":
        st.warning("该结果基于标题和摘要生成，不代表全文阅读。")
        with st.container(height=520):
            st.markdown(st.session_state.quick_readings.get(key, "暂无快速解读结果。"))
    elif result_type == "full":
        st.success("该结果基于 PDF 全文生成。")
        notes = st.session_state.full_reading_notes.get(key)
        if notes:
            with st.container(height=560):
                for title, body in split_markdown_sections(notes):
                    with st.container(border=True):
                        st.markdown(f"#### {title}")
                        if body:
                            st.markdown(body)
            if st.session_state.fulltext_debug.get(key):
                with st.expander("补全文调试信息", expanded=False):
                    st.json(st.session_state.fulltext_debug.get(key))
        else:
            st.info("暂无全文阅读笔记结果。")
    elif result_type == "full_error":
        st.error("自动读取失败，请切换到“阅读 PDF”页面手动上传 PDF。")
        with st.expander("错误详情", expanded=False):
            st.write(active.get("error") or "未知错误")
        if active.get("debug"):
            with st.expander("补全文调试信息", expanded=False):
                st.json(active.get("debug"))

    st.markdown('<div class="reading-dialog-actions">', unsafe_allow_html=True)
    action_cols = st.columns([1, 1, 4])
    if action_cols[0].button("关闭", key=f"close_reading_dialog_{result_type}_{key}"):
        close_reading_dialog()
    if result_type == "full" and key in st.session_state.full_reading_notes:
        action_cols[1].download_button(
            "下载 Markdown",
            data=st.session_state.full_reading_notes[key].encode("utf-8"),
            file_name=f"{key}_full_notes.md",
            mime="text/markdown",
            key=f"dialog_download_full_note_{key}",
        )
    if result_type == "full_error":
        action_cols[1].caption("请切换到“阅读 PDF”页面手动上传。")
    st.markdown("</div>", unsafe_allow_html=True)


def render_active_reading_dialog() -> None:
    active = st.session_state.get("active_reading_dialog")
    if not active:
        return

    title = active.get("title") or "论文"
    result_type = active.get("type")
    if result_type == "quick":
        dialog_title = f"快速解读：{title}"
    elif result_type == "full":
        dialog_title = f"全文阅读笔记：{title}"
    else:
        dialog_title = f"自动读取失败：{title}"

    inject_reading_dialog_css()

    if hasattr(st, "dialog"):
        @st.dialog(dialog_title, width="large")
        def reading_dialog():
            render_reading_dialog_body(active)

        reading_dialog()
        return

    st.markdown(
        """
        <style>
        .reading-float-box {
            border: 1px solid rgba(120, 120, 120, 0.35);
            border-radius: 12px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
            padding: 1rem;
            margin: 1rem 0;
            background: rgba(250, 250, 250, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(f"### {dialog_title}")
        render_reading_dialog_body(active)


def render_paper_card(index: int, paper: dict) -> None:
    verification = paper.get("verification", {})
    missing_fields = paper.get("missing_fields") or paper.get("verification", {}).get("missing_fields", [])
    pdf_url = paper_pdf_url(paper)
    paper_url = paper.get("url") or ""
    source_label = paper_display_source(paper)
    metadata_status = metadata_text(paper, verification)
    quality_status = quality_text(paper, verification)
    source_status = source_check_text(paper)
    hit_keywords = paper.get("hit_keywords") or []
    priority = display_priority(paper)

    with st.container(border=True):
        title = paper.get("title") or "API 未提供标题"
        st.markdown(f"### {index}. {title}")

        col1, col2, col3, col4 = st.columns(4)
        col1.write(f"**年份：** {paper.get('year') or 'API 未提供'}")
        col2.write(f"**来源：** {source_label}")
        col3.write(f"**Venue：** {paper.get('venue') or 'API 未提供'}")
        col4.write(f"**引用数：** {citation_display(paper)}")

        meta_col1, meta_col2, meta_col3 = st.columns(3)
        meta_col1.write(f"**PDF：** {pdf_status_display(paper)}")
        meta_col2.write(f"**正式发表状态：** {formal_publication_text(paper)}")
        meta_col3.write(f"**推荐优先级：** {priority}")
        render_compact_tags(compact_verification_tags(paper, source_status, metadata_status, quality_status))
        render_compact_tags(compact_dblp_tags(paper))
        note = publication_status_note(paper)
        if note:
            st.caption(note)

        st.markdown("**为什么推荐**")
        st.info(display_recommendation_reason(paper))

        st.markdown("**阅读建议**")
        st.write(display_reading_advice(paper))

        render_paper_actions(paper, pdf_url, paper_url)

        with st.expander("详细信息与核查", expanded=False):
            st.write(f"**作者：** {format_authors(paper)}")
            st.write(f"**英文 Abstract：** {paper.get('abstract') or 'API 未提供'}")
            st.write(f"**paperId：** {paper.get('paperId') or 'API 未提供'}")
            st.write(f"**来源命中：** {' / '.join(paper.get('sources') or []) or 'API 未提供'}")
            st.write(f"**arXiv ID：** {paper.get('arxiv_id') or '未提供'}")
            st.write(f"**DOI：** {paper.get('doi') or '未提供'}")
            st.write(f"**PDF 自动补全状态：** {paper.get('fulltext_status') or '未执行'}")
            if paper.get("fulltext_debug"):
                with st.expander("PDF 自动补全调试", expanded=False):
                    st.json(paper.get("fulltext_debug"))
            st.write(f"**缺失字段：** {', '.join(missing_fields) if missing_fields else '无'}")
            st.write(f"**命中关键词：** {'、'.join(hit_keywords[:8]) if hit_keywords else '暂无明确命中'}")
            st.markdown("**详细核查信息**")
            check_col1, check_col2, check_col3 = st.columns(3)
            check_col1.write(f"来源核查：{source_status}")
            check_col2.write(f"元数据：{metadata_status}")
            check_col3.write(f"质量判断：{quality_status}")
            dblp_col1, dblp_col2, dblp_col3 = st.columns(3)
            dblp_col1.write(f"DBLP 核查：{dblp_status_text(paper)}")
            dblp_col2.write(f"DBLP venue：{dblp_venue_text(paper)}")
            dblp_col3.write(f"正式发表状态：{formal_publication_text(paper)}")
            st.caption("质量判断不等于论文结论可靠，仅表示 venue 或来源层面的初步判断。")
            st.write(f"**推荐分数：** {paper.get('score', 0)}")
            st.write(f"**引用数：** {citation_display(paper)}")
            st.write(f"**Semantic Scholar citationCount：** {paper.get('source_metadata', {}).get('Semantic Scholar', {}).get('citationCount', '未提供')}")
            st.write(f"**原始 citationCount：** {paper.get('citationCount') if paper.get('citationCount') is not None else 'API 未提供'}")
            st.write(f"**DBLP status 原始值：** {paper.get('dblp_status') or 'not_checked'}")
            st.write(f"**DBLP title：** {paper.get('dblp_title') or '未找到'}")
            st.write(f"**DBLP year：** {paper.get('dblp_year') or '未找到'}")
            st.write(f"**DBLP url / ee：** {paper.get('dblp_url') or '未找到'}")
            if paper.get("dblp_error"):
                st.write(f"**DBLP 错误详情：** {paper.get('dblp_error')}")
            st.write(f"**source_metadata 来源：** {', '.join((paper.get('source_metadata') or {}).keys()) or '未提供'}")
            st.write(f"**技术来源：** {paper.get('_source') or 'API 未提供'}")


load_dotenv()

st.set_page_config(
    page_title="科研论文发现与阅读助手",
    page_icon="📄",
    layout="wide",
)

for key, default in {
    "notes": "",
    "active_file_id": "",
    "extracted_text": "",
    "cleaned_text": "",
    "pdf_processed": False,
    "discovery_result": None,
    "discovery_query": "",
    "last_query": "",
    "last_results": None,
    "last_search_status": "failed",
    "last_source": "尚未检索",
    "last_selected_sources": (),
    "last_search_mode": "",
    "selected_sources": ["arxiv", "semantic_scholar", "dblp"],
    "force_refresh": False,
    "current_source": "尚未检索",
    "current_mode": "尚未检索",
    "quick_readings": {},
    "full_reading_notes": {},
    "fulltext_lookup_cache": {},
    "fulltext_debug": {},
    "library_papers": {},
    "active_reading_dialog": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

sync_discovery_status()

st.title("面向研究生的科研论文发现与阅读助手")
st.caption("从研究方向出发，发现可信论文；从 PDF 出发，生成结构化阅读笔记。")
render_top_status_bar()

render_sidebar()

tab_discovery, tab_library, tab_notes = st.tabs(["发现论文", "我的文献库", "阅读 PDF"])

with tab_discovery:
    render_discovery_tab()

with tab_library:
    render_library_tab()

with tab_notes:
    render_pdf_notes_tab()

render_active_reading_dialog()
