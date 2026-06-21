import requests
import json
import os
import re
from datetime import datetime, timezone
import anthropic

# ── 설정 ──────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MAX_PAPERS = 10   # 하루 최대 수집 논문 수
DAYS_BACK  = 1    # 최근 며칠치 논문을 가져올지

def fetch_pubmed_ids():
    """PubMed에서 라모트리진 + Bipolar/Depression 논문 ID 목록 가져오기"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": (
            "(lamotrigine[Title/Abstract]) AND "
            "(bipolar[Title/Abstract] OR depression[Title/Abstract]) "
            "NOT epilepsy[Title/Abstract]"
        ),
        "sort": "date",
        "retmax": MAX_PAPERS,
        "retmode": "json",
        "datetype": "edat",
        "reldate": DAYS_BACK,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    print(f"[PubMed] 검색된 논문 수: {len(ids)}")
    return ids

def fetch_paper_details(pmid_list):
    """논문 ID → 상세 정보(제목, 저자, 초록, 날짜, 저널 등) 가져오기"""
    if not pmid_list:
        return []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmid_list),
        "retmode": "json",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    result = r.json().get("result", {})

    papers = []
    for pmid in pmid_list:
        item = result.get(pmid, {})
        if not item:
            continue

        # 저자 목록
        authors = [a.get("name", "") for a in item.get("authors", [])]

        # 게재일 파싱
        pub_date_str = item.get("pubdate", "")
        pub_date = parse_pubmed_date(pub_date_str)

        # Abstract 별도 API로 가져오기
        abstract = fetch_abstract(pmid)

        papers.append({
            "pmid":      pmid,
            "title":     item.get("title", "").rstrip("."),
            "authors":   authors,
            "journal":   item.get("source", ""),
            "pub_date":  pub_date,
            "volume":    item.get("volume", ""),
            "issue":     item.get("issue", ""),
            "pages":     item.get("pages", ""),
            "doi":       item.get("elocationid", "").replace("doi: ", ""),
            "abstract":  abstract,
        })

    return papers

def fetch_abstract(pmid):
    """PubMed에서 Abstract 텍스트만 가져오기"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        text = r.text
        # "Abstract\n" 다음 부분만 추출
        match = re.search(r"Abstract\n(.+?)(?:\n\nPMID|\Z)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"  Abstract 가져오기 실패 (PMID {pmid}): {e}")
    return ""

def parse_pubmed_date(date_str):
    """'2025 Jun 10' 또는 '2025' 같은 날짜 문자열 → 'YYYY년 MM월 DD일' 포맷"""
    date_str = date_str.strip()
    formats = ["%Y %b %d", "%Y %b", "%Y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if fmt == "%Y":
                return f"{dt.year}년"
            if fmt == "%Y %b":
                return f"{dt.year}년 {dt.month:02d}월"
            return f"{dt.year}년 {dt.month:02d}월 {dt.day:02d}일"
        except ValueError:
            continue
    return date_str  # 파싱 실패 시 원본 반환

def format_apa_citation(paper):
    """APA 형식 출처 생성"""
    authors = paper["authors"]
    if not authors:
        author_str = "Unknown"
    elif len(authors) == 1:
        author_str = authors[0]
    else:
        author_str = f"{authors[0]}, et al."

    year_match = re.match(r"(\d{4})", paper.get("pub_date", ""))
    year = year_match.group(1) if year_match else "n.d."

    journal = paper["journal"]
    volume  = paper["volume"]
    issue   = paper["issue"]
    pages   = paper["pages"]

    citation = f"{author_str} ({year}). {journal}"
    if volume:
        citation += f", {volume}"
    if issue:
        citation += f"({issue})"
    if pages:
        citation += f", {pages}"
    citation += "."
    return citation

def summarize_with_claude(paper):
    """Claude API로 제목 번역 + 2줄 요약 생성"""
    if not ANTHROPIC_API_KEY:
        return "API 키 없음", "요약 없음"

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    abstract_text = paper["abstract"] if paper["abstract"] else "(초록 없음)"

    prompt = f"""다음 의학 논문의 제목과 초록을 분석해 한국어로 답변해 주세요.

[논문 제목]
{paper['title']}

[초록]
{abstract_text}

다음 두 가지를 JSON 형식으로만 답변해 주세요. JSON 외 다른 텍스트는 절대 포함하지 마세요.

{{
  "title_ko": "제목을 자연스러운 한국어로 번역 (50자 이내)",
  "summary_ko": "연구의 핵심 내용을 한국어 두 문장으로 요약. 첫 문장은 연구 목적/방법, 두 번째 문장은 주요 결과/결론."
}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        return data.get("title_ko", "번역 실패"), data.get("summary_ko", "요약 실패")
    except Exception as e:
        print(f"  Claude API 오류: {e}")
        return "번역 실패", "요약 실패"

def load_existing_data():
    """기존에 저장된 논문 데이터 로드"""
    path = "docs/papers_data.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_data(papers):
    """논문 데이터 JSON으로 저장"""
    os.makedirs("docs", exist_ok=True)
    with open("docs/papers_data.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

def build_html(papers):
    """논문 데이터로 HTML 사이트 생성"""
    today = datetime.now(timezone.utc).strftime("%Y년 %m월 %d일")
    total = len(papers)

    cards_html = ""
    for p in papers:
        cards_html += f"""
    <article class="card">
      <div class="card-date">{p['pub_date']}</div>
      <h2 class="card-title-en">{p['title']}</h2>
      <h3 class="card-title-ko">🇰🇷 {p['title_ko']}</h3>
      <div class="card-summary">
        <span class="label">📋 연구 요약</span>
        <p>{p['summary_ko']}</p>
      </div>
      <div class="card-citation">
        <span class="label">📎 출처 (APA)</span>
        <p>{p['citation']}</p>
      </div>
      <a class="card-link" href="https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/" target="_blank">
        PubMed에서 원문 보기 →
      </a>
    </article>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>라모트리진 최신 연구논문</title>
  <style>
    :root {{
      --bg: #f0f4f8;
      --card: #ffffff;
      --primary: #2563eb;
      --primary-dark: #1e40af;
      --text: #1e293b;
      --muted: #64748b;
      --border: #e2e8f0;
      --accent: #eff6ff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.7;
    }}
    header {{
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      color: white;
      padding: 2.5rem 1.5rem;
      text-align: center;
    }}
    header h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }}
    header p  {{ font-size: 0.95rem; opacity: 0.85; }}
    .stats {{
      display: flex;
      justify-content: center;
      gap: 2rem;
      background: white;
      padding: 1rem;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .stats strong {{ color: var(--primary); font-size: 1.1rem; }}
    main {{
      max-width: 860px;
      margin: 2rem auto;
      padding: 0 1rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }}
    .card {{
      background: var(--card);
      border-radius: 12px;
      padding: 1.6rem;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
      border: 1px solid var(--border);
    }}
    .card-date {{
      font-size: 0.8rem;
      color: var(--primary);
      font-weight: 600;
      margin-bottom: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .card-title-en {{
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 0.4rem;
      line-height: 1.4;
    }}
    .card-title-ko {{
      font-size: 0.95rem;
      font-weight: 500;
      color: var(--muted);
      margin-bottom: 1rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
    }}
    .card-summary, .card-citation {{
      background: var(--accent);
      border-radius: 8px;
      padding: 0.9rem 1rem;
      margin-bottom: 0.8rem;
    }}
    .label {{
      display: block;
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--primary);
      margin-bottom: 0.35rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .card-summary p, .card-citation p {{
      font-size: 0.9rem;
      color: var(--text);
    }}
    .card-link {{
      display: inline-block;
      margin-top: 0.8rem;
      font-size: 0.85rem;
      color: var(--primary);
      text-decoration: none;
      font-weight: 600;
    }}
    .card-link:hover {{ text-decoration: underline; }}
    .no-papers {{
      text-align: center;
      padding: 4rem;
      color: var(--muted);
    }}
    footer {{
      text-align: center;
      padding: 2rem;
      font-size: 0.8rem;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>💊 라모트리진 최신 연구논문</h1>
    <p>Lamotrigine × Bipolar / Depression 관련 PubMed 최신 논문 자동 수집</p>
  </header>
  <div class="stats">
    <span>📅 마지막 업데이트: <strong>{today}</strong></span>
    <span>📄 누적 논문 수: <strong>{total}편</strong></span>
  </div>
  <main>
    {''.join([cards_html]) if cards_html else '<div class="no-papers"><p>오늘 새로운 논문이 없습니다. 내일 다시 확인해주세요.</p></div>'}
  </main>
  <footer>데이터 출처: PubMed (NCBI) | 번역·요약: Claude AI | 매일 자동 업데이트</footer>
</body>
</html>"""

    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("[HTML] index.html 생성 완료")

def main():
    print("=== 라모트리진 논문 수집 시작 ===")

    # 기존 데이터 로드
    existing = load_existing_data()
    existing_pmids = {p["pmid"] for p in existing}

    # 새 논문 ID 검색
    new_ids = [i for i in fetch_pubmed_ids() if i not in existing_pmids]
    print(f"[신규] {len(new_ids)}편 발견")

    # 상세 정보 가져오기
    new_papers_raw = fetch_paper_details(new_ids)

    # Claude로 번역·요약
    processed = []
    for i, p in enumerate(new_papers_raw, 1):
        print(f"  [{i}/{len(new_papers_raw)}] 처리 중: {p['title'][:60]}...")
        title_ko, summary_ko = summarize_with_claude(p)
        processed.append({
            **p,
            "title_ko":   title_ko,
            "summary_ko": summary_ko,
            "citation":   format_apa_citation(p),
        })

    # 최신순 정렬 후 저장
    all_papers = processed + existing
    save_data(all_papers)

    # HTML 생성
    build_html(all_papers)
    print("=== 완료 ===")

if __name__ == "__main__":
    main()
