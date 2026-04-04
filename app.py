import io
import re
import json
import math
import numpy as np
import pandas as pd
import streamlit as st

EXAMPLE_SURVEY = """
1.您的性别
A.男
B.女
2.您的年级
A.大一
B.大二
C.大三
3.您的生源地
A.城镇
B.农村
4.您是否担任班干部
A.是
B.否
5.您是否独生子女
A.独生子女
B.非独生子女
6.我有拖延症
A非常不符合
B不符合
C不确定
D符合
E非常符合
7.如果我有需要做的事，我会先做，而不是去做一些其他的事情
A非常不符合
B不符合
C不确定
D符合
E非常符合
"""

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except Exception:
    HAS_PYREADSTAT = False

st.set_page_config(page_title="问卷解析 + 维度/关系约束 + SPSS数据生成器（本地网页）", layout="wide")

# ---------- 可靠性分析函数 ----------
def cronbach_alpha(data):
    """计算Cronbach's α系数"""
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data).any(axis=1)]
    
    if data.shape[0] < 2 or data.shape[1] < 2:
        return np.nan
    
    k = data.shape[1]
    item_variances = np.var(data, axis=0, ddof=1)
    total_scores = np.sum(data, axis=1)
    total_variance = np.var(total_scores, ddof=1)
    
    if total_variance == 0:
        return np.nan
    
    alpha = (k / (k - 1)) * (1 - np.sum(item_variances) / total_variance)
    return alpha

def calculate_reliability_for_dimension(data, item_columns):
    """计算特定维度的可靠性"""
    if len(item_columns) < 2:
        return {"alpha": np.nan, "n_items": len(item_columns), "message": "题目数量不足"}
    
    dim_data = data[item_columns].dropna()
    
    if dim_data.shape[0] < 2:
        return {"alpha": np.nan, "n_items": len(item_columns), "message": "有效样本不足"}
    
    alpha = cronbach_alpha(dim_data)
    
    return {
        "alpha": alpha,
        "n_items": len(item_columns),
        "n_cases": dim_data.shape[0],
        "message": "计算成功"
    }
# ---------- 改进的基础函数 ----------

def parse_survey_text(txt: str):
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    questions = []
    i = 0
    q_pat = re.compile(r"^(\d+)\s*[\.、]\s*(.+)$")
    opt_pat = re.compile(r"^([A-E])\s*[\.、]?\s*(.+)$")
    cur = None
    while i < len(lines):
        m = q_pat.match(lines[i])
        if m:
            if cur:
                questions.append(cur)
            cur = {"qid": int(m.group(1)), "stem": m.group(2).strip(), "options": {}}
            i += 1
            while i < len(lines) and not q_pat.match(lines[i]):
                om = opt_pat.match(lines[i])
                if om and cur is not None:
                    cur["options"][om.group(1)] = om.group(2).strip()
                i += 1
            continue
        i += 1
    if cur:
        questions.append(cur)
    for q in questions:
        opts = " ".join(q["options"].values())
        if any(k in opts for k in ["非常不符合", "不符合", "不确定", "符合", "非常符合"]):
            q["scale_type"] = "likert_fit"
        elif any(k in opts for k in ["同意", "较同意", "较不同意", "不同意"]):
            q["scale_type"] = "likert_agree"
        else:
            q["scale_type"] = "categorical"
    return questions
def get_demo_questions(qs, scale_start_qid):
    """自动识别量表开始前的分类题，作为人口学变量"""
    demo_qs = []
    for q in qs:
        if q["qid"] < scale_start_qid and q.get("scale_type") == "categorical" and q.get("options"):
            demo_qs.append(q)
    return demo_qs


def normalize_percent_list(vals):
    arr = np.array(vals, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0
    if arr.sum() <= 0:
        arr = np.ones_like(arr)
    arr = arr / arr.sum() * 100.0
    arr = np.round(arr, 1)
    diff = round(100.0 - arr.sum(), 1)
    arr[-1] = round(arr[-1] + diff, 1)
    return arr.tolist()


def ensure_demo_config(cfg, qs):
    """根据当前问卷自动维护人口学配置"""
    scale_start_qid = int(cfg.get("scale_start_qid", 6))
    demo_qs = get_demo_questions(qs, scale_start_qid)

    demo = cfg.get("demo", {})
    enabled = demo.get("enabled", {})
    perc_map = demo.get("perc", {})

    new_enabled = {}
    new_perc_map = {}

    for q in demo_qs:
        qkey = f"Q{q['qid']}"
        n_opts = len(q["options"])

        old_enabled = enabled.get(qkey, True)
        old_perc = perc_map.get(qkey, None)

        if not isinstance(old_perc, list) or len(old_perc) != n_opts:
            base = round(100.0 / n_opts, 1)
            old_perc = [base] * n_opts
            old_perc[-1] = round(100.0 - sum(old_perc[:-1]), 1)

        new_enabled[qkey] = old_enabled
        new_perc_map[qkey] = normalize_percent_list(old_perc)

    cfg["demo"] = {
        "enabled": new_enabled,
        "perc": new_perc_map,
    }
    return cfg

def generate_latents(n, dim_names, corr_matrix=None, seed=42):
    """改进的潜变量生成，确保相关矩阵得到准确实现"""
    rng = np.random.default_rng(seed)
    k = len(dim_names)
    
    if corr_matrix is None:
        Z = rng.standard_normal(size=(n, k))
    else:
        R = np.array(corr_matrix, dtype=float)
        # 确保矩阵对称且正定
        R = (R + R.T) / 2.0  # 保证对称性
        np.fill_diagonal(R, 1.0)  # 保证对角线为1
        
        # 使用特征值分解确保正定性
        w, V = np.linalg.eigh(R)
        w = np.maximum(w, 1e-6)  # 确保所有特征值为正
        R_psd = V @ np.diag(w) @ V.T  # 重建正定矩阵
        
        # 使用Cholesky分解验证和修正
        try:
            L = np.linalg.cholesky(R_psd)
        except np.linalg.LinAlgError:
            # 如果Cholesky失败，使用SVD
            U, s, Vt = np.linalg.svd(R_psd)
            s = np.maximum(s, 1e-6)  # 修正特征值
            L = U @ np.diag(np.sqrt(s))  # 用SVD替代
        
        # 生成相关的标准正态变量
        Z_indep = rng.standard_normal(size=(n, k))
        Z = Z_indep @ L.T  # 将独立的标准正态变量转化为相关变量
    
    return pd.DataFrame(Z, columns=dim_names)


def apply_mediation(df_latents, A, C, B, a=0.6, b=0.6, cprime=0.1, seed=42):
    """改进的中介模型，确保路径系数更准确"""
    rng = np.random.default_rng(seed)
    n = len(df_latents)
    
    # 标准化A变量
    A_std = (df_latents[A] - df_latents[A].mean()) / df_latents[A].std()
    
    # C = a*A + e_C，控制误差项方差确保总方差为1
    var_eC = max(1e-6, 1.0 - a * a)
    eC = rng.standard_normal(n) * math.sqrt(var_eC)
    df_latents[C] = a * A_std + eC
    
    # 标准化C变量
    C_std = (df_latents[C] - df_latents[C].mean()) / df_latents[C].std()
    
    # B = b*C + c'*A + e_B
    var_struct = b * b + cprime * cprime + 2 * a * b * cprime
    var_struct = max(0.0, min(var_struct, 0.95))  # 限制在合理范围
    var_eB = max(1e-6, 1.0 - var_struct)
    
    eB = rng.standard_normal(n) * math.sqrt(var_eB)
    df_latents[B] = b * C_std + cprime * A_std + eB
    
    return df_latents
def apply_multi_predictors(df_latents, y_dim, x_dims, betas, seed=42):
    """
    一个因变量 y_dim，由多个自变量 x_dims 共同预测
    Y = b1*X1 + b2*X2 + ... + e

    参数：
    - y_dim: 因变量名（字符串）
    - x_dims: 自变量列表，例如 ["学习动机", "社会支持"]
    - betas: dict，例如 {"学习动机": 0.4, "社会支持": 0.3}
    """
    rng = np.random.default_rng(seed)

    if y_dim not in df_latents.columns:
        return df_latents

    valid_xs = [x for x in x_dims if x in df_latents.columns and x != y_dim]
    if not valid_xs:
        return df_latents

    # 标准化多个自变量
    X = df_latents[valid_xs].copy()
    X = (X - X.mean()) / (X.std(ddof=0) + 1e-8)

    beta_vec = np.array([float(betas.get(x, 0.0)) for x in valid_xs], dtype=float)

    # 多自变量线性组合
    y_pred = X.to_numpy() @ beta_vec

    # 控制结构方差，避免方差过大
    var_pred = np.var(y_pred)
    if var_pred > 0.95:
        y_pred = y_pred / np.sqrt(var_pred / 0.95)
        var_pred = np.var(y_pred)

    err_var = max(1e-6, 1.0 - var_pred)
    e = rng.standard_normal(len(df_latents)) * np.sqrt(err_var)

    df_latents[y_dim] = y_pred + e
    return df_latents

def latent_to_items(latent, n_items, mean=3.5, loading=0.60, noise=1.00, seed=42):
    """
    目的：让大维度在题目很多（15~40题）时，α 落在 0.80~0.93 的真实区间

    参数建议：
    - loading 建议在 0.5~0.7 之间（UI 那里可以默认 0.6）
    - noise   建议在 0.8~1.3 之间（UI 那里可以默认 1.0）
    """
    rng = np.random.default_rng(seed)
    n = len(latent)

    # 1) 标准化潜变量
    latent_std = (latent - latent.mean()) / (latent.std() + 1e-8)

    # 2) 每个题的载荷有一点差异，但整体不太高
    base_loading = loading          # 来自界面
    item_loadings = rng.normal(base_loading, 0.10, n_items)
    # clip 到 [0.30, 0.75]，避免过高的题间相关
    item_loadings = np.clip(item_loadings, 0.30, 0.75)

    # 3) 每道题有一个“难度偏移”（让均值不完全一样）
    item_diffs = rng.normal(0.0, 0.4, n_items)

    # 4) 噪声：完全由 noise 控制
    #    noise 越大，题目间相关越低，α 越低
    items = np.zeros((n, n_items), dtype=int)
    for j in range(n_items):
        true_score = item_loadings[j] * latent_std + item_diffs[j]
        error = rng.normal(0.0, noise, n)   # 这里是测量误差的主力

        continuous = mean + true_score + error

        # 四舍五入到 1~5
        items[:, j] = np.clip(np.rint(continuous), 1, 5).astype(int)

    return items


def make_spss_syntax_for_csv(csv_filename, df_cols, var_labels, value_labels):
    var_lines = []
    for col in df_cols:
        if col == "ID":
            var_lines.append(f"{col} F8.0")
        else:
            var_lines.append(f"{col} F3.0")
    var_block = " \n  ".join(var_lines)
    vl = "\n".join([f"VARIABLE LABELS {k} '{v}'." for k, v in var_labels.items() if k in df_cols])
    blocks = []
    for var, mapping in value_labels.items():
        if var not in df_cols:
            continue
        parts = [f"{k} '{v}'" for k, v in mapping.items()]
        blocks.append(f"VALUE LABELS {var}\n  " + "\n  ".join(parts) + ".")
    vlab = "\n\n".join(blocks)
    return f"""* Auto-generated by Survey Synth WebApp.
GET DATA
  /TYPE=TXT
  /FILE='{csv_filename}'
  /ENCODING='UTF8'
  /DELCASE=LINE
  /DELIMITERS=","
  /QUALIFIER='"'
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /VARIABLES=
  {var_block}.
EXECUTE.

{vl}

{vlab}

EXECUTE.
"""


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def generate_data_with_subdims(cfg, qs):
    """
    统一的数据生成函数：
    - 每一道题 Qk 都是 1–5 的整数
    - 大维度均值 = 这些题目的真·算术平均值（和EXCEL AVERAGE完全一样）
    - 小维度均值 = 小维度题目的真·算术平均值（同样是EXCEL AVERAGE）
    - 人口学效应 / 中介 / 维度相关都只在潜变量 Z 上控制，不再对大/小维度均值做“二次加工”
    """
    dims_map = cfg.get("dimensions", {})
    dim_names = list(dims_map.keys())
    if not dim_names:
        raise ValueError("dimensions 为空，先在第2页配置大维度。")

    # ---------- 基本参数 ----------
    N = int(cfg.get("N", 630))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    all_qids = sorted([q["qid"] for q in qs])
    min_qid = min(all_qids)
    max_qid = max(all_qids)
    scale_start_qid = int(cfg.get("scale_start_qid", 6))
    scale_start_qid = max(min_qid, min(max_qid, scale_start_qid))

    # ---------- 1) 人口学变量（自动按问卷分类题生成） ----------
    cfg = ensure_demo_config(cfg, qs)
    demo = cfg.get("demo", {})
    demo_qs = get_demo_questions(qs, scale_start_qid)

    demo_values = {}
    demo_scores = {}

    for q in demo_qs:
        qid = q["qid"]
        qkey = f"Q{qid}"

        if not demo.get("enabled", {}).get(qkey, True):
            continue

        n_opts = len(q["options"])
        perc = demo.get("perc", {}).get(qkey, None)

        if not isinstance(perc, list) or len(perc) != n_opts:
            base = round(100.0 / n_opts, 1)
            perc = [base] * n_opts
            perc[-1] = round(100.0 - sum(perc[:-1]), 1)

        perc = normalize_percent_list(perc)
        probs = np.array(perc, dtype=float)
        probs = probs / probs.sum()

        cats = rng.choice(np.arange(1, n_opts + 1), size=N, p=probs)
        demo_values[qkey] = cats

        if n_opts == 1:
            score = np.zeros(N)
        elif n_opts == 2:
            score = np.where(cats == 2, 1.0, -1.0)
        else:
            score = ((cats.astype(float) - 1) / (n_opts - 1) - 0.5) * 2.0

        demo_scores[qkey] = score

    # ---------- 2) 潜变量 Z（含相关矩阵） ----------
    k = len(dim_names)
    R = np.eye(k)
    cm = cfg.get("corr_matrix")
    if (isinstance(cm, list) and len(cm) == k
            and all(isinstance(r, list) and len(r) == k for r in cm)):
        try:
            R = np.array(cm, dtype=float)
        except Exception:
            R = np.eye(k)

    # 利用你已有的 generate_latents
    Z = generate_latents(N, dim_names, corr_matrix=R, seed=seed)

    # ---------- 3) 应用多自变量 → 单因变量结构（可选） ----------
    multi_reg = cfg.get("multi_regression")
    if multi_reg:
        y_dim = multi_reg.get("y")
        x_dims = multi_reg.get("xs", [])
        betas = multi_reg.get("betas", {})

        if y_dim in Z.columns and isinstance(x_dims, list) and len(x_dims) > 0:
            Z = apply_multi_predictors(
                Z,
                y_dim=y_dim,
                x_dims=x_dims,
                betas=betas,
                seed=seed + 7,
            )

    # ---------- 4) 人口学差异 β 叠加在 Z 上 ----------
    demo_effects = cfg.get("demo_effects", {})
    for d in dim_names:
        eff = demo_effects.get(d, {})
        if not isinstance(eff, dict):
            eff = {}
        b_gender = float(eff.get("gender", 0.0) or 0.0)
        b_grade = float(eff.get("grade", 0.0) or 0.0)
        b_origin = float(eff.get("origin", 0.0) or 0.0)
        b_cadre = float(eff.get("cadre", 0.0) or 0.0)
        b_only = float(eff.get("only", 0.0) or 0.0)

        if any(abs(x) > 0 for x in [b_gender, b_grade, b_origin, b_cadre, b_only]):
            delta = (
                b_gender * demo_scores.get("Q1", np.zeros(N))
                + b_grade * demo_scores.get("Q2", np.zeros(N))
                + b_origin * demo_scores.get("Q3", np.zeros(N))
                + b_cadre * demo_scores.get("Q4", np.zeros(N))
                + b_only * demo_scores.get("Q5", np.zeros(N))
            )
            Z[d] = Z[d] + delta

    # ---------- 5) 生成题目数据 Qk（1–5分） ----------
    out = pd.DataFrame({"ID": np.arange(1, N + 1)})

    # 人口学变量写入
    for qkey, cats in demo_values.items():
        out[qkey] = cats

    # 题目归属映射
    qid_to_dim = {}
    for d in dim_names:
        for qid in dims_map[d]:
            qid_to_dim[qid] = d

    rev = set(cfg.get("reverse_items", []))
    item_params = cfg.get("item_params", {})
    item_mean = float(item_params.get("mean", 3.6))
    item_loading = float(item_params.get("loading", 0.60))
    item_noise = float(item_params.get("noise", 1.00))

    # 遍历大维度，生成本维度所有题目
    for d in dim_names:
        qids = sorted([qid for qid, dd in qid_to_dim.items()
                       if dd == d and qid >= scale_start_qid])
        if not qids:
            continue

        disc = latent_to_items(
            Z[d].to_numpy(), len(qids),
            mean=item_mean, loading=item_loading, noise=item_noise,
            seed=seed + 13 + (hash(d) % 1000),
        )
        for idx, qid in enumerate(qids):
            x = disc[:, idx]
            if qid in rev:
                x = 6 - x  # 反向计分
            out[f"Q{qid}"] = x

    # ---------- 6) 大维度均值（真·算术平均） ----------
    cols = ["ID"]
    for qid in all_qids:
        col = f"Q{qid}"
        if col in out.columns:
            cols.append(col)

    for d in dim_names:
        qcols = [f"Q{qid}" for qid in dims_map[d]
                 if f"Q{qid}" in out.columns and qid >= scale_start_qid]
        if qcols:
            out[f"{d}_mean"] = out[qcols].mean(axis=1)
            cols.append(f"{d}_mean")

    # ---------- 7)（取消）小维度均值 ----------
    # 现在不再用 cfg["subdimensions"] 生成额外的小维度均分，
    # 每个“小维度”直接作为一个维度写在 cfg["dimensions"] 里即可。
    # 如果想自己在 SPSS 里再拼总分，可以后处理。

    # 最终列顺序
    out = out[cols]
    return out
# ---------- session_state 初始化 ----------
if "questions" not in st.session_state:
    st.session_state.questions = []
if "config" not in st.session_state:
    st.session_state.config = {}
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

st.title("问卷解析 + 维度/关系约束 + SPSS数据生成器（本地网页）")
tabs = st.tabs(
    [
        "1) 导入/解析问卷",
        "2) 维度与计分 & 人口学",
        "3) 关系约束（人口学差异/相关/中介）",
        "4) 生成与导出",
        "5) 📊 可靠性分析",
    ]
)

# ---------- Tab 1 ----------
with tabs[0]:
    st.subheader("粘贴问卷文本 → 自动识别题号/题干/选项")
    raw = st.text_area("问卷文本", value=st.session_state.get("raw_text", ""), height=320)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("解析问卷", type="primary", use_container_width=True):
            st.session_state.questions = parse_survey_text(raw)
            st.session_state.raw_text = raw
            st.success(f"已解析 {len(st.session_state.questions)} 道题。")
    with c2:
        if st.button("载入示例", use_container_width=True):
            st.session_state.raw_text = EXAMPLE_SURVEY
            st.session_state.questions = parse_survey_text(EXAMPLE_SURVEY)
            st.success(f"已载入示例并解析 {len(st.session_state.questions)} 道题。")

    if st.session_state.questions:
        dfq = pd.DataFrame(
            [
                {
                    "qid": q["qid"],
                    "scale_type": q["scale_type"],
                    "stem": q["stem"],
                    "options": " | ".join([f"{k}:{v}" for k, v in q["options"].items()]),
                }
                for q in st.session_state.questions
            ]
        ).sort_values("qid")
        st.dataframe(dfq, use_container_width=True)


# ---------- Tab 2 ----------
with tabs[1]:
    st.subheader("配置：维度归属、反向题、人口学 & 题目分布")
    qs = st.session_state.questions
    if not qs:
        st.info("先到第 1 页解析问卷。")
    else:
        all_qids = sorted([q["qid"] for q in qs])
        min_qid = min(all_qids)
        max_qid = max(all_qids)

        if not st.session_state.config:
            default_start = 6 if any(q >= 6 for q in all_qids) else min_qid
            st.session_state.config = {
                "N": 630,
                "seed": 42,
                "scale_start_qid": default_start,
                "reverse_items": [],
                "dimensions": {
                    "A_dim": [q for q in all_qids if default_start <= q < default_start + 10],
                },
                "subdimensions": {},
                "demo": {
                    "enabled": {},
                    "perc": {},                    
                },
                "item_params": {"mean": 3.6, "loading": 0.80, "noise": 1.0},  # 改进的默认参数
                "corr_matrix": None,
                "multi_regression": {
                   "y": "A_dim",
                   "xs": [],
                   "betas": {},
                },
                "demo_effects": {
                    "A_dim": {"gender": 0.8, "grade": 0.3, "origin": 0.0, "cadre": 0.0, "only": 0.0},  # 增强效应
                },
            }

        cfg = st.session_state.config

        c1, c2 = st.columns([1, 1])
        with c1:
            cfg["N"] = st.number_input("样本量 N", 30, 5000, int(cfg.get("N", 630)), 10)
        with c2:
            cfg["seed"] = st.number_input("随机种子 seed", 0, 10000, int(cfg.get("seed", 42)), 1)

        default_start_qid = int(cfg.get("scale_start_qid", 6))
        default_start_qid = max(min_qid, min(max_qid, default_start_qid))
        scale_start_qid = st.number_input(
            "量表维度起始题号", min_value=min_qid, max_value=max_qid,
            value=default_start_qid, step=1,
        )
        cfg["scale_start_qid"] = int(scale_start_qid)
        cfg = ensure_demo_config(cfg, qs)
        scale_qids = [qid for qid in all_qids if qid >= cfg["scale_start_qid"]]

        # --- 维度设置 ---
        st.markdown("### 大维度设置")
        dims_dict = cfg.get("dimensions", {})
        dims_items = list(dims_dict.items())

        if st.button("➕ 新增维度"):
            base_name = "新维度"
            idx = 1
            new_name = base_name
            existing_names = set(dims_dict.keys())
            while new_name in existing_names:
                new_name = f"{base_name}{idx}"
                idx += 1
            dims_dict[new_name] = []
            cfg["dimensions"] = dims_dict
            st.session_state.config = cfg
            st.rerun()

        delete_keys = []
        for i, (orig_name, qid_list) in enumerate(dims_items):
            with st.expander(f"维度 {i+1}：{orig_name}", expanded=True):
                st.text_input("维度名称", value=orig_name, key=f"dim_name_{i}")
                valid_default = [qid for qid in qid_list if qid in scale_qids]
                st.multiselect("包含题目（可多选）", options=scale_qids, default=valid_default, key=f"dim_items_{i}")
                                # 实时显示题目数量和可靠性预估
                selected_items = st.session_state.get(f"dim_items_{i}", valid_default)
                if len(selected_items) > 0:
                    reliability_est = min(0.95, 0.4 + 0.05 * len(selected_items))
                    
                    if len(selected_items) >= 6:
                        st.success(f"✅ {len(selected_items)}个题目，预估α≈{reliability_est:.2f}")
                    elif len(selected_items) >= 4:
                        st.warning(f"⚠️ {len(selected_items)}个题目，预估α≈{reliability_est:.2f}，建议增加到6个")
                    else:
                        st.error(f"❌ {len(selected_items)}个题目，可靠性可能不足，强烈建议增加题目")
                if st.button("删除本维度", key=f"dim_del_{i}"):
                    delete_keys.append(orig_name)

        if delete_keys:
            for k in delete_keys:
                if k in dims_dict:
                    dims_dict.pop(k)
            cfg["dimensions"] = dims_dict
            st.session_state.config = cfg
            st.rerun()
        else:
            new_dims = {}
            for i, (orig_name, qid_list) in enumerate(dims_items):
                name = st.session_state.get(f"dim_name_{i}", "").strip() or orig_name
                items = st.session_state.get(f"dim_items_{i}", None)
                if items is None:
                    items = qid_list
                if name:
                    new_dims[name] = sorted(set(items))
            cfg["dimensions"] = new_dims


        # --- 人口学变量设置（自动按问卷分类题生成） ---
        st.markdown("### 人口学变量设置")
        cfg = ensure_demo_config(cfg, qs)
        demo = cfg.get("demo", {})
        demo_qs = get_demo_questions(qs, cfg["scale_start_qid"])

        if not demo_qs:
            st.info("当前未识别到人口学分类题。请把人口学题放在量表题之前。")
        else:
            st.caption("系统会自动识别量表开始前的分类题作为人口学变量。你只需要调整比例。")

            for q in demo_qs:
                qid = q["qid"]
                qkey = f"Q{qid}"
                options = list(q["options"].values())
                n_opts = len(options)

                if qkey not in demo["enabled"]:
                    demo["enabled"][qkey] = True
                if qkey not in demo["perc"] or len(demo["perc"][qkey]) != n_opts:
                    base = round(100.0 / n_opts, 1)
                    tmp = [base] * n_opts
                    tmp[-1] = round(100.0 - sum(tmp[:-1]), 1)
                    demo["perc"][qkey] = tmp

                with st.expander(f"Q{qid}：{q['stem']}", expanded=True):
                    demo["enabled"][qkey] = st.checkbox(
                        "启用该人口学变量",
                        value=demo["enabled"].get(qkey, True),
                        key=f"enable_{qkey}"
                    )

                    st.markdown("**比例设置：**")
                    cols = st.columns(n_opts)
                    current = demo["perc"][qkey]

                    new_vals = []
                    running = 0.0
                    for i, opt_text in enumerate(options):
                        if i < n_opts - 1:
                            with cols[i]:
                                v = st.number_input(
                                    f"{opt_text}(%)",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=float(current[i]),
                                    step=1.0,
                                    key=f"{qkey}_opt_{i}"
                                )
                            max_allowed = max(0.0, 100.0 - running)
                            v = min(v, max_allowed)
                            new_vals.append(v)
                            running += v
                        else:
                            last_val = round(max(0.0, 100.0 - running), 1)
                            with cols[i]:
                                st.metric(f"{opt_text}(%)", f"{last_val:.1f}")
                            new_vals.append(last_val)

                    demo["perc"][qkey] = normalize_percent_list(new_vals)

                    preview_df = pd.DataFrame({
                        "编码": list(range(1, n_opts + 1)),
                        "选项": options,
                        "比例(%)": demo["perc"][qkey]
                    })
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)

            cfg["demo"] = demo
        # --- 题目参数设置（改进的默认值） ---
        st.markdown("### 题目参数设置")
        st.caption("💡 提示：载荷越高、噪声越低，可靠性越高；题目多时适当把噪声调大，可以避免 α 过高。")

        item_params = cfg.get("item_params", {})
        col1, col2, col3 = st.columns(3)

        with col1:
            item_params["mean"] = st.number_input(
                "题目均值",
                1.0, 5.0,
                float(item_params.get("mean", 3.6)),
                0.1,
            )

        with col2:
            item_params["loading"] = st.number_input(
                "因子载荷（建议 0.5~0.7）",
                0.3, 0.9,
                float(item_params.get("loading", 0.60)),   # 默认 0.60
                0.05,
            )

        with col3:
            item_params["noise"] = st.number_input(
                "噪声水平（建议 0.8~1.3）",
                0.3, 1.5,
                float(item_params.get("noise", 1.00)),     # 默认 1.00
                0.05,
            )

        cfg["item_params"] = item_params

        # 反向题设置
        rev_txt = st.text_input(
            "反向题题号（逗号分隔）",
            value=",".join(map(str, cfg.get("reverse_items", []))) if cfg.get("reverse_items") else "",
            help="例如：8,12,15 表示这些题目在 1~5 上按 1↔5 反向计分。",
        )
        cfg["reverse_items"] = [
            int(x.strip()) for x in rev_txt.split(",") if x.strip().isdigit()
        ]

        if st.button("保存当前配置", type="primary"):
            st.session_state.config = cfg
            st.success("配置已保存！")

       # --- JSON 配置编辑器 ---
        st.markdown("---")
        st.markdown("### JSON 配置编辑器")
        st.caption("高级用户可以直接编辑 JSON 配置，或复制配置用于备份")
        config_json = json.dumps(st.session_state.config, indent=2, ensure_ascii=False)
        edited_config_json = st.text_area("编辑 JSON 配置", config_json, height=400, key="json_editor")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("应用 JSON 配置", use_container_width=True):
                try:
                    parsed = json.loads(edited_config_json)
                    if "config" in parsed and isinstance(parsed["config"], dict):
                        new_config = parsed["config"]
                        if "questions" in parsed and isinstance(parsed["questions"], list):
                            st.session_state.questions = parsed["questions"]
                        if "raw_text" in parsed:
                            st.session_state.raw_text = parsed["raw_text"]
                    else:
                        new_config = parsed
                    required_fields = ["N", "seed", "scale_start_qid", "dimensions", "demo", "item_params"]
                    missing = [f for f in required_fields if f not in new_config]
                    if missing:
                        st.error(f"配置缺少必要字段: {', '.join(missing)}")
                    elif not isinstance(new_config["dimensions"], dict) or not new_config["dimensions"]:
                        st.error("dimensions 不能为空字典")
                    else:
                        demo_obj = new_config.get("demo", {})
                        demo_obj.setdefault("enabled", {})
                        demo_obj.setdefault("perc", {})
                        new_config["demo"] = demo_obj
                        st.session_state.config = new_config
                        st.success("✅ JSON 配置已成功应用！")
                        st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON 格式错误：{e}")
                except Exception as e:
                    st.error(f"❌ 应用配置时出错：{e}")
        with col2:
            st.download_button(
                "导出当前配置 JSON",
                data=json.dumps(st.session_state.config, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="survey_config.json",
                mime="application/json",
                use_container_width=True,
            )

# ---------- Tab 3 ----------
with tabs[2]:
    st.subheader("关系约束：各人口学差异・维度相关矩阵・中介模型")
    cfg = st.session_state.config
    qs = st.session_state.questions
    if not cfg or not qs:
        st.info("先完成第 1–2 页。")
    else:
        dims = list(cfg.get("dimensions", {}).keys())
        if len(dims) < 1:
            st.warning("维度数量不足，请先在第 2 页配置。")
        else:
            st.markdown("### 人口学差异（按维度设置 β）")
            st.caption(
                "💡 建议：β 在 ±0.3 到 ±1.2 之间效果较好。\n"
                "性别β：女生更高>0；年级β：高年级更高>0；生源地β：农村生源更高>0；班干部β：班干部更高>0；独生β：独生子女更高>0。"
            )

            demo_effects = cfg.get("demo_effects")
            if not isinstance(demo_effects, dict):
                demo_effects = {}
            for d in dims:
                demo_effects.setdefault(d, {})
                demo_effects[d].setdefault("gender", 0.8)  # 增强默认效应
                demo_effects[d].setdefault("grade", 0.3)
                demo_effects[d].setdefault("origin", 0.0)
                demo_effects[d].setdefault("cadre", 0.0)
                demo_effects[d].setdefault("only", 0.0)

            rows = []
            for d in dims:
                eff = demo_effects.get(d, {})
                rows.append({
                    "维度": d,
                    "性别β(女高>0,男高<0)": float(eff.get("gender", 0.8)),
                    "年级β(高年级高>0)": float(eff.get("grade", 0.3)),
                    "生源地β(农村高>0)": float(eff.get("origin", 0.0)),
                    "班干部β(班干部高>0)": float(eff.get("cadre", 0.0)),
                    "独生β(独生高>0)": float(eff.get("only", 0.0)),
                })
            df_demo = pd.DataFrame(rows)
            df_demo_edit = st.data_editor(df_demo, num_rows="fixed", use_container_width=True, hide_index=True)

            new_demo_effects = {}
            for _, row in df_demo_edit.iterrows():
                dim_name = row["维度"]
                def _safe(v):
                    try:
                        return float(v)
                    except Exception:
                        return 0.0
                new_demo_effects[dim_name] = {
                    "gender": _safe(row["性别β(女高>0,男高<0)"]),
                    "grade": _safe(row["年级β(高年级高>0)"]),
                    "origin": _safe(row["生源地β(农村高>0)"]),
                    "cadre": _safe(row["班干部β(班干部高>0)"]),
                    "only": _safe(row["独生β(独生高>0)"]),
                }
            cfg["demo_effects"] = new_demo_effects

            st.markdown("### 维度相关矩阵（任意两个维度之间可设相关）")
            st.caption("💡 建议：相关系数在 ±0.3 到 ±0.7 之间比较合理。对角线固定为 1。")

            k = len(dims)
            cm = cfg.get("corr_matrix")
            if (isinstance(cm, list) and len(cm) == k
                    and all(isinstance(r, list) and len(r) == k for r in cm)):
                try:
                    base_mat = np.array(cm, dtype=float)
                except Exception:
                    base_mat = np.eye(k)
            else:
                base_mat = np.eye(k)

            df_corr = pd.DataFrame(base_mat, index=dims, columns=dims)
            df_corr_edit = st.data_editor(df_corr, num_rows="fixed", use_container_width=True, hide_index=True)

            M = df_corr_edit.values.astype(float)
            for i in range(k):
                M[i, i] = 1.0
            M = (M + M.T) / 2.0  # 确保矩阵是对称的
            np.fill_diagonal(M, 1.0)  # 强制将对角线设为1
            cfg["corr_matrix"] = M.tolist()

            st.markdown("### 多自变量 → 单因变量")
            st.caption("💡 选择一个因变量 Y，多个自变量 X 可自由增减；每个自变量的回归系数 β 可分别设置。")

            multi_reg = cfg.get("multi_regression", {})

            y_default = multi_reg.get("y", dims[0])
            if y_default not in dims:
                y_default = dims[0]

            y_dim = st.selectbox(
                "因变量 Y（只能选一个）",
                dims,
                index=dims.index(y_default),
                key="multi_y"
            )

            x_options = [d for d in dims if d != y_dim]
            x_default = multi_reg.get("xs", [])
            x_default = [x for x in x_default if x in x_options]

            x_dims = st.multiselect(
                "自变量 X（可多选，可自己增加/减少）",
                options=x_options,
                default=x_default,
                key="multi_xs"
            )

            beta_dict = multi_reg.get("betas", {})

            if x_dims:
                st.markdown("#### 各自变量对因变量的回归系数 β")

                beta_rows = []
                for x in x_dims:
                    beta_rows.append({
                        "自变量": x,
                        "回归系数β": float(beta_dict.get(x, 0.30))
                    })

                df_beta = pd.DataFrame(beta_rows)
                df_beta_edit = st.data_editor(
                    df_beta,
                    num_rows="fixed",
                    use_container_width=True,
                    hide_index=True
                )

                new_betas = {}
                for _, row in df_beta_edit.iterrows():
                    x_name = row["自变量"]
                    try:
                        new_betas[x_name] = float(row["回归系数β"])
                    except Exception:
                        new_betas[x_name] = 0.0
            else:
                st.info("请至少选择 1 个自变量。")
                new_betas = {}

            cfg["multi_regression"] = {
                "y": y_dim,
                "xs": x_dims,
                "betas": new_betas
            }

            st.session_state.config = cfg
            st.success("已保存多自变量 → 单因变量关系约束。")


# ---------- Tab 4 ----------
with tabs[3]:
    st.subheader("生成并导出（CSV / .sps / .sav）")
    cfg = st.session_state.config
    qs = st.session_state.questions
    if not cfg or not qs:
        st.info("先完成前面三步。")
    else:
        dims_map = cfg.get("dimensions", {})
        dim_names = list(dims_map.keys())
        if not dim_names:
            st.warning("还没有设置任何维度，请先到第 2 页配置。")
        else:
            # 这里只负责调用统一生成函数
            if st.button("生成数据", type="primary"):
                out = generate_data_with_subdims(cfg, qs)
                st.session_state.generated = out
                st.success(
                    f"✅ 已生成 {out.shape[0]} 行 × {out.shape[1]} 列数据（小维度 = 题目真·算术平均值）"
                )
                # 如果你后面想在这里继续做“实时可靠性检查”，
                # 也可以接着用 out 这个变量来算；注意不要再用 N 这个名字。
                st.markdown("### 📊 实时可靠性检查")
                reliability_results = {}
                all_qualified = True
                
                reliability_summary = []
                for d in dim_names:
                    qcols = [f"Q{qid}" for qid in dims_map[d] if f"Q{qid}" in out.columns]
                    if len(qcols) >= 2:
                        result = calculate_reliability_for_dimension(out, qcols)
                        reliability_results[d] = result
                        
                        alpha = result["alpha"]
                        n_items = result["n_items"]
                        
                        if not np.isnan(alpha):
                            if alpha >= 0.75:
                                status = "✅ 达标"
                            else:
                                status = "❌ 不足"
                                all_qualified = False
                            
                            reliability_summary.append({
                                "维度": d,
                                "题目数": n_items,
                                "Cronbach's α": f"{alpha:.3f}",
                                "状态": status
                            })
                        else:
                            reliability_summary.append({
                                "维度": d,
                                "题目数": n_items,
                                "Cronbach's α": "计算失败",
                                "状态": "❌ 错误"
                            })
                            all_qualified = False
                
                if reliability_summary:
                    summary_df = pd.DataFrame(reliability_summary)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    qualified_count = sum(1 for r in reliability_summary if "✅" in r["状态"])
                    total_count = len(reliability_summary)
                    
                    if all_qualified and reliability_results:
                        st.balloons()
                        st.success(f"🎉 恭喜！所有 {total_count} 个维度可靠性均达标！")
                    else:
                        st.warning(f"⚠️ {qualified_count}/{total_count} 个维度达标")

            # 在按钮块外展示结果
            if "generated" in st.session_state:
                out = st.session_state.generated
                cfg = st.session_state.config
                demo = cfg.get("demo", {})

                st.dataframe(out.head(50), use_container_width=True)

                st.markdown("### 人口学差异显著性（基于本次模拟数据）")
                dim_mean_cols = [c for c in out.columns if c.endswith("_mean")]

                def _ttest_binary(y, g01):
                    y = np.asarray(y, dtype=float)
                    g01 = np.asarray(g01, dtype=int)
                    mask0, mask1 = g01 == 0, g01 == 1
                    n0, n1 = mask0.sum(), mask1.sum()
                    if n0 < 2 or n1 < 2:
                        return None, None, None
                    y0, y1 = y[mask0], y[mask1]
                    m0, m1 = y0.mean(), y1.mean()
                    v0, v1 = y0.var(ddof=1), y1.var(ddof=1)
                    sp2 = ((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2)
                    if sp2 <= 0:
                        return m1 - m0, None, None
                    t = (m1 - m0) / math.sqrt(sp2 * (1.0 / n0 + 1.0 / n1))
                    p = 2 * (1.0 - _norm_cdf(abs(t)))
                    return m1 - m0, t, p

                def _reg_slope(y, x):
                    y = np.asarray(y, dtype=float)
                    x = np.asarray(x, dtype=float)
                    n = len(y)
                    if n < 3:
                        return None, None, None
                    xm, ym = x.mean(), y.mean()
                    Sxx = ((x - xm) ** 2).sum()
                    if Sxx <= 0:
                        return None, None, None
                    b1 = ((x - xm) * (y - ym)).sum() / Sxx
                    resid = y - (ym + b1 * (x - xm))
                    s2 = (resid ** 2).sum() / (n - 2)
                    if s2 <= 0:
                        return b1, None, None
                    t = b1 / math.sqrt(s2 / Sxx)
                    p = 2 * (1.0 - _norm_cdf(abs(t)))
                    return b1, t, p

                results = []
                for dim_col in dim_mean_cols:
                    y = out[dim_col].to_numpy()
                    if "Q1" in out.columns and demo.get("use_Q1", False):
                        g = (out["Q1"].to_numpy() == 2).astype(int)
                        diff, t, p = _ttest_binary(y, g)
                        if p is not None:
                            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                            results.append([dim_col, "性别(女 vs 男)", diff, t, p, sig])
                    if "Q2" in out.columns and demo.get("use_Q2", False):
                        x = out["Q2"].to_numpy().astype(float)
                        b1, t, p = _reg_slope(y, x)
                        if p is not None:
                            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                            results.append([dim_col, "年级(高年级更高为正)", b1, t, p, sig])
                    if "Q3" in out.columns and demo.get("use_Q3", False):
                        g = (out["Q3"].to_numpy() == 2).astype(int)
                        diff, t, p = _ttest_binary(y, g)
                        if p is not None:
                            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                            results.append([dim_col, "生源地(农村 vs 城镇)", diff, t, p, sig])
                    if "Q4" in out.columns and demo.get("use_Q4", False):
                        g = (out["Q4"].to_numpy() == 1).astype(int)
                        diff, t, p = _ttest_binary(y, g)
                        if p is not None:
                            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                            results.append([dim_col, "班干部(是 vs 否)", diff, t, p, sig])
                    if "Q5" in out.columns and demo.get("use_Q5", False):
                        g = (out["Q5"].to_numpy() == 1).astype(int)
                        diff, t, p = _ttest_binary(y, g)
                        if p is not None:
                            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                            results.append([dim_col, "独生(独生 vs 非独生)", diff, t, p, sig])

                if results:
                    df_sig = pd.DataFrame(
                        results,
                        columns=["维度均分变量", "人口学变量", "差异/斜率(高-低)", "t值", "p(正态近似)", "显著性"],
                    )
                    df_sig["差异/斜率(高-低)"] = df_sig["差异/斜率(高-低)"].round(3)
                    df_sig["t值"] = df_sig["t值"].round(3)
                    df_sig["p(正态近似)"] = df_sig["p(正态近似)"].round(4)
                    st.dataframe(df_sig, use_container_width=True)
                    st.caption("显著性说明：*** p<.001，** p<.01，* p<.05，ns 不显著（基于正态近似）。")
                else:
                    st.info("当前没有可检验的人口学变量或维度均分列。")

                # 变量标签
                var_labels = {"ID": "Respondent ID"}
                for q in qs:
                    col = f"Q{q['qid']}"
                    if col in out.columns:
                        var_labels[col] = q["stem"][:240]
                for d in dim_names:
                    if f"{d}_mean" in out.columns:
                        var_labels[f"{d}_mean"] = f"{d}（大维度均分）"

                subdims_all = cfg.get("subdimensions", {})
                if isinstance(subdims_all, dict):
                    for big_dim, subdict in subdims_all.items():
                        if not isinstance(subdict, dict):
                            continue
                        for sub_name, sub_qids in subdict.items():
                            if not sub_qids:
                                continue
                            safe_sub = re.sub(r"\W+", "", sub_name)
                            col_name = f"{big_dim}_{safe_sub}_mean"
                            if col_name in out.columns:
                                var_labels[col_name] = f"{big_dim}-{sub_name}（小维度均分）"

                # 值标签
                value_labels = {}

                demo_qs = get_demo_questions(qs, cfg.get("scale_start_qid", 6))
                for q in demo_qs:
                    col = f"Q{q['qid']}"
                    if col in out.columns:
                        opts = list(q["options"].values())
                        value_labels[col] = {i + 1: opts[i] for i in range(len(opts))}

                for qid in all_qids:
                    col = f"Q{qid}"
                    if col in out.columns and qid >= scale_start_qid:
                        value_labels[col] = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}

                # 下载按钮
                csv_bytes = out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    "下载 CSV",
                    data=csv_bytes,
                    file_name="synthetic_data.csv",
                    mime="text/csv",
                )

                sps = make_spss_syntax_for_csv(
                    "synthetic_data.csv", out.columns.tolist(), var_labels, value_labels,
                )
                st.download_button(
                    "下载 .sps（导入 + 标签）",
                    data=sps.encode("utf-8"),
                    file_name="import_and_label.sps",
                    mime="text/plain",
                )

                if HAS_PYREADSTAT:
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tf:
                        path = tf.name
                    try:
                        pyreadstat.write_sav(
                            out, path,
                            column_labels=var_labels,
                            variable_value_labels=value_labels,
                        )
                        with open(path, "rb") as f:
                            sav = f.read()
                        st.download_button(
                            "下载 .sav（含标签）",
                            data=sav,
                            file_name="synthetic_data.sav",
                            mime="application/octet-stream",
                        )
                    finally:
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                else:
                    st.info("未安装 pyreadstat：可以用 CSV + .sps 在 SPSS 中导入，效果相同。")


# ---------- Tab 5 - 可靠性分析 ----------
with tabs[4]:
    st.subheader("📊 详细可靠性分析")
    
    if "generated" not in st.session_state:
        st.info("请先到第4页生成数据")
    else:
        out = st.session_state.generated
        cfg = st.session_state.config
        dims_map = cfg.get("dimensions", {})
        
        if not dims_map:
            st.warning("没有找到维度配置")
        else:
            st.markdown("### 🎯 各维度详细分析")
            
            overall_summary = []
            
            for dim_name, qids in dims_map.items():
                qcols = [f"Q{qid}" for qid in qids if f"Q{qid}" in out.columns]
                
                if len(qcols) < 2:
                    st.warning(f"维度「{dim_name}」题目不足")
                    continue
                
                result = calculate_reliability_for_dimension(out, qcols)
                alpha = result["alpha"]
                
                if np.isnan(alpha):
                    st.error(f"维度「{dim_name}」计算失败")
                    continue
                
                # 判断等级
                if alpha >= 0.9:
                    level = "🟢 优秀"
                elif alpha >= 0.8:
                    level = "🟢 良好"
                elif alpha >= 0.75:
                    level = "🟡 可接受"
                else:
                    level = "🔴 不足"
                
                overall_summary.append({
                    "维度": dim_name,
                    "题目数": len(qcols),
                    "Cronbach's α": f"{alpha:.3f}",
                    "等级": level
                })
                
                # 详细分析
                with st.expander(f"📋 {dim_name} - α = {alpha:.3f}", expanded=alpha < 0.75):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Cronbach's α", f"{alpha:.3f}")
                    with col2:
                        st.metric("题目数量", len(qcols))
                    with col3:
                        st.metric("有效样本", result["n_cases"])
                    
                    # 改进建议
                    if alpha < 0.75:
                        st.error("⚠️ **改进建议：**")
                        suggestions = []
                        
                        if len(qcols) < 6:
                            suggestions.append(f"• 增加题目数量（当前{len(qcols)}个，建议至少6个）")
                        
                        current_params = cfg.get("item_params", {})
                        current_loading = current_params.get("loading", 0.85)
                        current_noise = current_params.get("noise", 0.45)
                        
                        if current_loading < 0.85:
                            suggestions.append(f"• 提高因子载荷（当前{current_loading}，建议≥0.85）")
                        
                        if current_noise > 0.4:
                            suggestions.append(f"• 降低噪声水平（当前{current_noise}，建议≤0.4）")
                        
                        suggestions.append("• 重新生成数据并检查结果")
                        
                        for suggestion in suggestions:
                            st.markdown(suggestion)
                    else:
                        st.success("✅ 该维度可靠性达标！")
            
            # 汇总表
            if overall_summary:
                st.markdown("### 📈 可靠性汇总")
                summary_df = pd.DataFrame(overall_summary)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # 达标统计
                total_dims = len(overall_summary)
                qualified_dims = sum(1 for item in overall_summary 
                                   if float(item["Cronbach's α"]) >= 0.75)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总维度数", total_dims)
                with col2:
                    st.metric("达标维度", qualified_dims)
                with col3:
                    st.metric("达标率", f"{qualified_dims/total_dims*100:.1f}%")
                
                if qualified_dims == total_dims:
                    st.balloons()
                    st.success("🎉 恭喜！所有维度可靠性均达到0.75以上！")
                else:
                    st.warning(f"还有 {total_dims - qualified_dims} 个维度需要改进")
            
            # 快速改进按钮
            st.markdown("### ⚡ 快速改进")
            if st.button("🔧 应用高可靠性参数", use_container_width=True):
                cfg["item_params"]["loading"] = 0.78
                cfg["item_params"]["noise"] = 0.5
                cfg["item_params"]["mean"] = 3.5
                st.session_state.config = cfg
                st.success("✅ 已应用高可靠性参数，请返回第4页重新生成数据")
