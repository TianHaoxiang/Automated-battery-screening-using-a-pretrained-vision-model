import re
import json
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# 公开仪表板（Cycle Test Cell List）token（来自 Battery Archive 公共链接）
DASHBOARD_URL = "https://database.batteryarchive.org/public/dashboards/dG5C30gXNzH77rMeApZ4lIOrjy0m7pencFScWFA5"

OUT_XLSX = "dataset_info.xlsx"

def looks_like_cell_list(payload: dict) -> bool:
    """
    Redash 常见返回结构中，真正的数据通常在 data.rows / query_result.data.rows 等位置。
    我们用“是否包含 cell_id/Cell ID/ah/cathode 等列”来判断是否是 cell list。
    """
    def extract_rows_and_cols(obj):
        # 尝试多种 Redash 结构
        candidates = []
        if isinstance(obj, dict):
            # 常见：{"query_result": {"data": {"rows": [...], "columns": [...]}}}
            qr = obj.get("query_result")
            if isinstance(qr, dict):
                data = qr.get("data", {})
                rows = data.get("rows")
                cols = data.get("columns")
                candidates.append((rows, cols))
            # 也可能：{"data": {"rows": [...], "columns": [...]}}
            data = obj.get("data")
            if isinstance(data, dict):
                candidates.append((data.get("rows"), data.get("columns")))
        return candidates

    for rows, cols in extract_rows_and_cols(payload):
        if isinstance(rows, list) and len(rows) > 0:
            # rows 是 list[dict]
            if isinstance(rows[0], dict):
                keys = set(k.lower() for k in rows[0].keys())
                if ("cell id" in keys) or ("cell_id" in keys) or ("cellid" in keys):
                    return True
                # 兜底：同时出现这些典型字段
                if ("cathode" in keys) and ("ah" in keys):
                    return True
    return False

def extract_rows(payload: dict) -> list[dict]:
    # 按常见 Redash 结构取 rows
    if isinstance(payload, dict):
        if isinstance(payload.get("query_result"), dict):
            data = payload["query_result"].get("data", {})
            rows = data.get("rows")
            if isinstance(rows, list):
                return rows
        if isinstance(payload.get("data"), dict):
            rows = payload["data"].get("rows")
            if isinstance(rows, list):
                return rows
    return []

def main():
    captured = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def on_response(resp):
            try:
                ct = resp.headers.get("content-type", "")
                if "application/json" not in ct:
                    return
                data = resp.json()
                if looks_like_cell_list(data):
                    captured.append(data)
            except Exception:
                return

        page.on("response", on_response)

        page.goto(DASHBOARD_URL, wait_until="networkidle", timeout=120_000)

        # 有的 Redash 仪表板会延迟发起查询；再多等一会儿
        try:
            page.wait_for_timeout(10_000)
        except PWTimeout:
            pass

        browser.close()

    if not captured:
        raise RuntimeError(
            "未捕获到 cell list 的 JSON 返回。可能原因：页面加载策略变化/需要更长等待/接口非 JSON。\n"
            "建议：用浏览器开发者工具 Network 里确认表格请求的返回类型，并把请求 URL 发我。"
        )

    # 多个候选时，取 rows 最大的那个（最可能是“全量列表”）
    best = max(captured, key=lambda d: len(extract_rows(d)))
    rows = extract_rows(best)

    df = pd.DataFrame(rows)

    # 统一 Cell ID 字段名
    # 可能是 "Cell ID" 或 "cell_id"
    if "Cell ID" not in df.columns and "cell_id" in df.columns:
        df.rename(columns={"cell_id": "Cell ID"}, inplace=True)

    # 你的要求：Ah 记录为“标况/标称容量”
    # 可能列名是 "ah" 或 "Ah"
    if "ah" in df.columns:
        df.rename(columns={"ah": "Ah (标称容量/标况容量)"}, inplace=True)
    elif "Ah" in df.columns:
        df.rename(columns={"Ah": "Ah (标称容量/标况容量)"}, inplace=True)

    # 只要你当前最核心的字段，也可以把其它列都保留（建议保留，后续有用）
    # 这里保留全部列，同时把关键列放前面
    key_order = [c for c in ["Cell ID", "Ah (标称容量/标况容量)"] if c in df.columns]
    remaining = [c for c in df.columns if c not in key_order]
    df = df[key_order + remaining]

    # 导出 Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cells")

    print(f"OK: 导出 {len(df)} 条电芯样本到 {OUT_XLSX}")

if __name__ == "__main__":
    main()
