"""
表格存储服务 — Excel/CSV 结构化查询通道

与文档不同，表格数据走 pandas 直接查询，不经过分块和向量检索。
Agent 可以用自然语言查询表格，底层自动转成 pandas 操作。

存储结构:
  { "file_name": {"df": DataFrame, "columns": [...], "row_count": N, ...} }
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

log = logger.bind(module="table_store")


class TableStore:
    """表格仓库，数据持久化到磁盘 (parquet)，重启不丢失"""

    DATA_DIR = "./table_data"

    def __init__(self) -> None:
        self._tables: dict[str, dict[str, Any]] = {}
        self._load_from_disk()

    # ── Persistence ─────────────────────────────────────────

    def _save_to_disk(self, name: str) -> None:
        import os
        os.makedirs(self.DATA_DIR, exist_ok=True)
        entry = self._tables.get(name)
        if entry and entry["df"] is not None:
            safe_name = name.replace("/", "_").replace("\\", "_")
            path = os.path.join(self.DATA_DIR, f"{safe_name}.parquet")
            entry["df"].to_parquet(path, index=False)
            log.debug(f"表格已持久化: {path}")

    def _load_from_disk(self) -> None:
        import os, glob
        os.makedirs(self.DATA_DIR, exist_ok=True)
        for path in glob.glob(os.path.join(self.DATA_DIR, "*.parquet")):
            try:
                df = pd.read_parquet(path)
                name = os.path.basename(path).replace(".parquet", "")
                # 还原原始文件名（去除安全替换）
                self._tables[name] = {
                    "df": df,
                    "meta": {
                        "name": name, "source": path,
                        "columns": list(df.columns), "row_count": len(df),
                        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                    },
                }
                log.info(f"从磁盘恢复表格: {name} ({len(df)} rows)")
            except Exception as e:
                log.warning(f"恢复表格失败: {path} → {e}")

    # ── CRUD ────────────────────────────────────────────────

    def add_table(self, name: str, df: pd.DataFrame, source: str = "") -> dict:
        """存入一张表，返回元信息"""
        # 清洗：去掉全空行/列
        df = df.dropna(how="all").dropna(axis=1, how="all")
        # 统一列名为字符串
        df.columns = [str(c).strip() for c in df.columns]

        info = {
            "name": name,
            "source": source,
            "columns": list(df.columns),
            "row_count": len(df),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        }
        self._tables[name] = {"df": df, "meta": info}
        self._save_to_disk(name)
        log.info(f"表格入库: {name} → {len(df)} rows, {len(df.columns)} cols")
        return info

    def get_table(self, name: str) -> pd.DataFrame | None:
        entry = self._tables.get(name)
        return entry["df"].copy() if entry else None

    def list_tables(self) -> list[dict]:
        return [entry["meta"] for entry in self._tables.values()]

    def remove_table(self, name: str) -> bool:
        if name in self._tables:
            del self._tables[name]
            import os
            safe_name = name.replace("/", "_").replace("\\", "_")
            path = os.path.join(self.DATA_DIR, f"{safe_name}.parquet")
            if os.path.exists(path):
                os.remove(path)
            return True
        return False

    # ── Query ───────────────────────────────────────────────

    def query(self, table_name: str, operation: str, **kwargs) -> str:
        """
        执行表格查询操作。

        支持的操作:
          - head / tail: 预览前/后 N 行
          - filter: 按列值筛选 (column=xxx, value=xxx)
          - search: 在所有列中搜索关键词
          - stats: 数值列统计 (mean, max, min, count)
          - columns: 列出列名
          - sample: 随机采样 N 行
        """
        df = self.get_table(table_name)
        if df is None:
            available = list(self._tables.keys())
            return f"表格 '{table_name}' 不存在。可用表格: {available}"

        try:
            if operation == "head":
                n = int(kwargs.get("n", 10))
                return df.head(n).to_string(index=False)

            elif operation == "tail":
                n = int(kwargs.get("n", 10))
                return df.tail(n).to_string(index=False)

            elif operation == "columns":
                return (f"列名: {list(df.columns)}\n行数: {len(df)}\n"
                        f"前5行预览:\n{df.head(5).to_string(index=False)}")

            elif operation == "filter":
                col = kwargs.get("column", "")
                val = kwargs.get("value", "")
                if col not in df.columns:
                    close = [c for c in df.columns if val.lower() in c.lower() or col.lower() in c.lower()]
                    hint = f" 相似列: {close}" if close else ""
                    return f"列 '{col}' 不存在。可用列: {list(df.columns)}.{hint}"
                # 模糊匹配（字符串列用 contains，其他用 ==）
                if df[col].dtype == object:
                    mask = df[col].astype(str).str.contains(str(val), case=False, na=False)
                else:
                    try:
                        mask = df[col] == type(df[col].iloc[0])(val)
                    except (ValueError, TypeError):
                        mask = df[col].astype(str).str.contains(str(val), case=False, na=False)
                result = df[mask]
                return f"找到 {len(result)} 行:\n{result.to_string(index=False)}"

            elif operation == "search":
                keyword = str(kwargs.get("keyword", ""))
                if not keyword:
                    return "请提供 search 关键词 (keyword=xxx)"
                # 在所有列中搜索（不限于字符串列）
                mask = pd.Series(False, index=df.index)
                for col in df.columns:
                    try:
                        mask |= df[col].astype(str).str.contains(keyword, case=False, na=False)
                    except Exception:
                        pass
                result = df[mask]
                if len(result) == 0:
                    return (f"搜索 '{keyword}' 找到 0 行。\n"
                            f"表格列名: {list(df.columns)}\n"
                            f"前5行样本:\n{df.head(5).to_string(index=False)}\n"
                            f"提示: 请根据列名和样本调整搜索词重试。")
                return f"搜索 '{keyword}' 找到 {len(result)} 行:\n{result.to_string(index=False)}"

            elif operation == "stats":
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) == 0:
                    return "没有数值列可以统计"
                stats = df[numeric_cols].describe().to_string()
                return f"数值列统计:\n{stats}"

            elif operation == "sample":
                n = min(int(kwargs.get("n", 5)), len(df))
                return df.sample(n).to_string(index=False)

            else:
                return f"不支持的操作: '{operation}'。支持: head, tail, filter, search, stats, columns, sample"

        except Exception as e:
            return f"查询失败: {e}"
