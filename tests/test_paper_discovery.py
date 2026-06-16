import unittest
from unittest.mock import Mock, patch

from paper_discovery import (
    build_source_warning,
    calculate_recommendation_priority,
    can_reuse_discovery_result,
    determine_search_status,
    discover_papers,
    enrich_paper_with_dblp,
    expand_keywords_with_deepseek,
    generate_reading_advice,
    hit_keywords,
    merge_and_deduplicate_by_title,
    merge_papers,
    normalize_dblp_paper,
    normalize_semantic_scholar_paper,
    search_dblp_publications,
    score_paper,
    search_semantic_scholar,
    sort_papers,
    verify_paper,
    verify_with_dblp,
)


class PaperDiscoveryTests(unittest.TestCase):
    def test_verify_paper_statuses(self):
        complete = {
            "title": "A Survey on Watermarking for Generative Models",
            "authors": [{"name": "Alice"}],
            "year": 2024,
            "venue": "ACL",
            "abstract": "This paper studies watermarking.",
            "citationCount": 10,
            "url": "https://www.semanticscholar.org/paper/example",
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            "paperId": "abc123",
        }
        partial = {
            "title": "Watermarking Large Language Models",
            "authors": [],
            "year": 2023,
            "url": "https://www.semanticscholar.org/paper/example2",
            "paperId": "def456",
        }
        unverified = {
            "authors": [{"name": "Bob"}],
            "year": 2022,
            "paperId": "ghi789",
        }

        self.assertEqual(verify_paper(complete)["status"], "verified")
        self.assertEqual(verify_paper(partial)["status"], "partial")
        self.assertEqual(verify_paper(unverified)["status"], "unverified")
        self.assertEqual(verify_paper(complete)["metadata_verified"], "完整")
        self.assertEqual(verify_paper(complete)["source_verified"], "来源可追溯")
        self.assertIn(verify_paper(complete)["quality_verified"], {"顶会/顶刊", "待进一步确认"})

    def test_arxiv_quality_is_preprint_not_top_venue(self):
        paper = {
            "title": "AIGC watermarking",
            "authors": [{"name": "Alice"}],
            "year": 2025,
            "venue": "arXiv",
            "abstract": "A watermarking preprint.",
            "citationCount": 0,
            "url": "https://arxiv.org/abs/2501.00001",
            "openAccessPdf": {"url": "https://arxiv.org/pdf/2501.00001"},
            "paperId": "arxiv:2501.00001",
            "_source": "arXiv fallback",
        }

        verified = verify_paper(paper)

        self.assertEqual(verified["source_label"], "arXiv")
        self.assertEqual(verified["quality_verified"], "arXiv 预印本")
        self.assertIn("预印本", generate_reading_advice(verified))

    def test_recommendation_priority_uses_metadata_score_and_keywords(self):
        high_priority = {
            "title": "LLM Watermarking for AI-generated Content",
            "abstract": "This paper studies model watermarking.",
            "year": 2025,
            "venue": "NeurIPS",
            "citationCount": 120,
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            "hit_keywords": ["LLM watermarking"],
            "score": 12,
        }
        low_priority = {
            "title": "Unrelated Old Paper",
            "abstract": "",
            "year": 2015,
            "venue": "",
            "citationCount": 0,
            "score": 1,
        }

        self.assertEqual(calculate_recommendation_priority(high_priority), "高")
        self.assertEqual(calculate_recommendation_priority(low_priority), "低")

    @patch.dict("os.environ", {}, clear=True)
    def test_public_semantic_scholar_limit_warning_is_merged(self):
        warning = build_source_warning(
            source="arXiv fallback",
            auth_mode="public",
            warning="Semantic Scholar 未认证模式受到限流，已切换到 arXiv fallback。",
        )

        self.assertEqual(
            warning,
            "当前 Semantic Scholar 未配置 API Key，系统已尝试未认证检索；由于公共限流，已自动切换到 arXiv 预印本数据源。",
        )

    def test_mock_fallback_is_not_reusable_cache_result(self):
        mock_result = {
            "source": "mock fallback",
            "papers": [{"title": "Demo Paper", "_source": "Mock fallback"}],
        }
        real_result = {
            "source": "arXiv fallback",
            "papers": [{"title": "Real Preprint", "_source": "arXiv fallback"}],
        }

        self.assertEqual(determine_search_status(mock_result), "fallback_mock")
        self.assertFalse(
            can_reuse_discovery_result(
                current_query="AIGC模型水印",
                last_query="AIGC模型水印",
                last_result=mock_result,
                last_search_status="fallback_mock",
                last_source="mock fallback",
                force_refresh=False,
            )
        )
        self.assertTrue(
            can_reuse_discovery_result(
                current_query="AIGC模型水印",
                last_query="AIGC模型水印",
                last_result=real_result,
                last_search_status="success",
                last_source="arXiv fallback",
                force_refresh=False,
            )
        )
        self.assertFalse(
            can_reuse_discovery_result(
                current_query="AIGC模型水印",
                last_query="AIGC模型水印",
                last_result=real_result,
                last_search_status="success",
                last_source="arXiv fallback",
                force_refresh=True,
            )
        )

    def test_sort_papers_prefers_scored_results(self):
        papers = [
            {"title": "Old", "year": 2020, "citationCount": 200, "openAccessPdf": None, "_search_rank": 2},
            {"title": "Recent PDF", "year": 2024, "citationCount": 5, "openAccessPdf": {"url": "x"}, "_search_rank": 1},
            {"title": "Most Relevant", "year": 2023, "citationCount": 1, "openAccessPdf": None, "_search_rank": 0},
            {"title": "Recent Cited", "year": 2024, "citationCount": 100, "openAccessPdf": None, "_search_rank": 1},
        ]

        sorted_titles = [paper["title"] for paper in sort_papers(papers)]

        self.assertEqual(sorted_titles[0], "Recent PDF")
        self.assertIn("Recent Cited", sorted_titles[:3])

    def test_merge_deduplicates_by_lowercase_title(self):
        papers = [
            {"title": "LLM Watermarking", "paperId": "1"},
            {"title": "llm watermarking", "paperId": "2"},
            {"title": "Diffusion Watermarking", "paperId": "3"},
        ]

        merged = merge_and_deduplicate_by_title(papers)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["paperId"], "1")

    def test_merge_papers_combines_sources_and_prefers_dblp_venue(self):
        semantic = normalize_semantic_scholar_paper(
            {
                "paperId": "s2-1",
                "title": "Watermarking Large Language Models",
                "authors": [{"name": "Alice"}],
                "year": 2024,
                "venue": "arXiv",
                "abstract": "Semantic abstract.",
                "citationCount": 42,
                "url": "https://semanticscholar.org/paper/s2-1",
                "openAccessPdf": {"url": "https://example.com/s2.pdf"},
            }
        )
        dblp = normalize_dblp_paper(
            {
                "title": "Watermarking Large Language Models",
                "authors": ["Alice", "Bob"],
                "year": 2024,
                "venue": "ACL",
                "url": "https://dblp.org/rec/conf/acl/example",
                "ee": "https://doi.org/10.0000/example",
            }
        )

        merged = merge_papers([semantic, dblp])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["sources"], ["semantic_scholar", "dblp"])
        self.assertEqual(merged[0]["venue"], "ACL")
        self.assertEqual(merged[0]["citation_count"], 42)
        self.assertEqual(merged[0]["publication_status"], "confirmed")

    def test_score_paper_uses_keywords_pdf_citations_and_venue(self):
        paper = {
            "title": "LLM Watermarking for AI-generated Content",
            "abstract": "This paper studies model watermarking.",
            "year": 2025,
            "citationCount": 150,
            "venue": "NeurIPS",
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        }

        score = score_paper(paper, ["LLM watermarking", "AI-generated content watermarking"])

        self.assertGreaterEqual(score, 8)
        self.assertIn("LLM watermarking", hit_keywords(paper, ["LLM watermarking"]))

    @patch("paper_discovery.requests.get")
    def test_search_dblp_publications_parses_official_json(self, mock_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "result": {
                "hits": {
                    "hit": [
                        {
                            "info": {
                                "title": "Watermarking Large Language Models.",
                                "authors": {"author": [{"text": "Alice"}, {"text": "Bob"}]},
                                "year": "2024",
                                "venue": "ACL",
                                "url": "https://dblp.org/rec/conf/acl/example",
                                "ee": "https://doi.org/10.0000/example",
                            }
                        }
                    ]
                }
            }
        }
        mock_get.return_value = response

        results = search_dblp_publications("Watermarking Large Language Models", limit=5)

        self.assertEqual(results[0]["title"], "Watermarking Large Language Models")
        self.assertEqual(results[0]["authors"], ["Alice", "Bob"])
        self.assertEqual(results[0]["year"], 2024)
        self.assertEqual(results[0]["venue"], "ACL")

    @patch("paper_discovery.search_dblp_publications")
    def test_enrich_paper_with_dblp_marks_formal_publication(self, mock_search):
        mock_search.return_value = [
            {
                "title": "Watermarking Large Language Models",
                "authors": ["Alice"],
                "year": 2024,
                "venue": "ACL",
                "url": "https://dblp.org/rec/conf/acl/example",
                "ee": "https://doi.org/10.0000/example",
            }
        ]
        paper = {
            "title": "Watermarking Large Language Models",
            "authors": [{"name": "Alice"}],
            "year": 2024,
            "venue": "arXiv",
            "abstract": "A paper about watermarking.",
            "citationCount": 0,
            "url": "https://arxiv.org/abs/2401.00001",
            "paperId": "arxiv:2401.00001",
            "_source": "arXiv fallback",
        }

        enriched = enrich_paper_with_dblp(verify_paper(paper))
        alias_enriched = verify_with_dblp(verify_paper(paper))

        self.assertEqual(enriched["dblp_status"], "found")
        self.assertEqual(alias_enriched["dblp_status"], "found")
        self.assertEqual(enriched["dblp_venue"], "ACL")
        self.assertEqual(enriched["publication_status"], "confirmed")
        self.assertEqual(enriched["quality_judgement"], "正式发表，venue 已确认")
        self.assertEqual(enriched["dblp_verification"]["status"], "found")

    @patch("paper_discovery.search_dblp_publications")
    def test_enrich_paper_with_dblp_failure_does_not_break_flow(self, mock_search):
        mock_search.side_effect = RuntimeError("dblp unavailable")
        paper = verify_paper(
            {
                "title": "Watermarking Large Language Models",
                "authors": [{"name": "Alice"}],
                "year": 2024,
                "venue": "arXiv",
                "abstract": "A paper about watermarking.",
                "citationCount": 0,
                "url": "https://arxiv.org/abs/2401.00001",
                "paperId": "arxiv:2401.00001",
                "_source": "arXiv fallback",
            }
        )

        enriched = enrich_paper_with_dblp(paper)

        self.assertEqual(enriched["dblp_status"], "error")
        self.assertEqual(enriched["publication_status"], "pending")
        self.assertIn("dblp unavailable", enriched["dblp_error"])

    @patch("paper_discovery.search_dblp_publications")
    def test_mock_fallback_skips_dblp_verification(self, mock_search):
        paper = verify_paper(
            {
                "title": "Demo Paper",
                "authors": [{"name": "Alice"}],
                "year": 2024,
                "venue": "Demo",
                "abstract": "Mock paper.",
                "citationCount": 0,
                "url": "https://example.com/demo",
                "paperId": "mock-1",
                "_source": "Mock fallback",
            }
        )

        enriched = enrich_paper_with_dblp(paper)

        mock_search.assert_not_called()
        self.assertEqual(enriched["dblp_status"], "not_checked")
        self.assertEqual(enriched["publication_status"], "pending")

    @patch.dict("os.environ", {}, clear=True)
    def test_keyword_expansion_without_deepseek_key_uses_local_fallback(self):
        expanded = expand_keywords_with_deepseek("AIGC模型水印")

        self.assertEqual(expanded["keyword_expansion_source"], "local fallback")
        self.assertIn("LLM watermarking", expanded["search_queries"])

    @patch.dict("os.environ", {}, clear=True)
    @patch("paper_discovery.requests.get")
    def test_semantic_scholar_without_api_key_uses_empty_headers(self, mock_get):
        response = Mock()
        response.json.return_value = {"data": []}
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        search_semantic_scholar("watermarking", limit=3)

        self.assertEqual(mock_get.call_args.kwargs["headers"], {})

    @patch("paper_discovery.expand_keywords_with_deepseek")
    @patch("paper_discovery.search_semantic_scholar")
    @patch("paper_discovery.search_arxiv")
    @patch("paper_discovery.search_dblp_publications")
    def test_discover_papers_falls_back_to_arxiv(self, mock_dblp, mock_arxiv, mock_semantic, mock_expand):
        mock_expand.return_value = {"keywords": ["watermarking"], "search_queries": ["watermarking"]}
        mock_semantic.side_effect = RuntimeError("semantic scholar unavailable")
        mock_dblp.return_value = []
        mock_arxiv.return_value = [
            {
                "title": "Watermarking Generative Models",
                "authors": [{"name": "Alice"}],
                "year": 2024,
                "venue": "arXiv",
                "abstract": "A paper about watermarking.",
                "citationCount": 0,
                "url": "https://arxiv.org/abs/2401.00001",
                "openAccessPdf": {"url": "https://arxiv.org/pdf/2401.00001"},
                "paperId": "arxiv:2401.00001",
            }
        ]

        result = discover_papers("AIGC模型水印", limit=10)

        self.assertEqual(result["source"], "arXiv fallback")
        self.assertEqual(result["papers"][0]["title"], "Watermarking Generative Models")
        self.assertFalse(result["debug_info"]["mock_fallback"])
        self.assertTrue(result["debug_info"]["sources"]["arxiv"]["called"])
        self.assertEqual(result["debug_info"]["sources"]["arxiv"]["count"], 1)

    @patch("paper_discovery.expand_keywords_with_deepseek")
    @patch("paper_discovery.search_semantic_scholar")
    @patch("paper_discovery.search_arxiv")
    @patch("paper_discovery.search_dblp_publications")
    def test_discover_papers_reports_429_fallback(self, mock_dblp, mock_arxiv, mock_semantic, mock_expand):
        mock_expand.return_value = {"keywords": ["watermarking"], "search_queries": ["watermarking"]}
        mock_semantic.side_effect = RuntimeError("429 Too Many Requests")
        mock_dblp.return_value = []
        mock_arxiv.return_value = []

        result = discover_papers("AIGC模型水印", limit=10)

        self.assertEqual(result["source"], "mock fallback")
        self.assertEqual(result["debug_info"]["mock_fallback_reason"], "所有真实数据源返回 0 条")
        self.assertIn("限流", result["warning"])


if __name__ == "__main__":
    unittest.main()
