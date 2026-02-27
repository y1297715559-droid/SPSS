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


def generate_latents(n, dim_names, corr_matrix=None, seed=42):
    """改进的潜变量生成，确保相关矩阵得到准确实现"""
    rng = np.random.default_rng(seed)
    k = len(dim_names)
    
    if corr_matrix is None:
        Z = rng.standard_normal(size=(n, k))
    else:
        R = np.array(corr_matrix, dtype=float)
        # 确保矩阵对称且正定
        R = (R + R.T) / 2.0
        np.fill_diagonal(R, 1.0)
        
        # 使用特征值分解确保正定性
        w, V = np.linalg.eigh(R)
        w = np.maximum(w, 1e-6)  # 确保所有特征值为正
        R_psd = V @ np.diag(w) @ V.T
        
        # Cholesky分解
        try:
            L = np.linalg.cholesky(R_psd)
        except np.linalg.LinAlgError:
            # 如果Cholesky失败，使用SVD
            U, s, Vt = np.linalg.svd(R_psd)
            s = np.maximum(s, 1e-6)
            L = U @ np.diag(np.sqrt(s))
        
        # 生成相关的标准正态变量
        Z_indep = rng.standard_normal(size=(n, k))
        Z = Z_indep @ L.T
    
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


def latent_to_items(latent, n_items, mean=3.5, loading=0.85, noise=0.38, seed=42):
    """改进的题目生成函数，专门优化可靠性"""
    rng = np.random.default_rng(seed)
    n = len(latent)
    
    # 标准化潜变量
    latent_std = (latent - latent.mean()) / (latent.std() + 1e-8)
    
    # 关键改进1: 适度的题目间差异
    item_difficulties = rng.normal(0, 0.18, n_items)  # 控制难度差异
    
    # 关键改进2: 适度的因子载荷变异
    base_loading = max(0.80, loading)  # 确保最低0.80
    item_loadings = rng.normal(base_loading, 0.03, n_items)  # 适度变异
    item_loadings = np.clip(item_loadings, 0.75, 0.92)  # 控制在0.75-0.92
    
    # 关键改进3: 适度的共同因子载荷
    common_factor = rng.standard_normal(n)
    common_loading = 0.12  # 适中的共同因子
    
    # 生成题目分数
    items = np.zeros((n, n_items))
    for i in range(n_items):
        # 真分数 = 主因子载荷 * 潜变量 + 共同因子 + 难度参数
        true_score = (item_loadings[i] * latent_std + 
                     common_loading * common_factor + 
                     item_difficulties[i])
        
        # 关键改进4: 适度的测量误差
        error_var = max(0.06, 1.0 - item_loadings[i] ** 2 - common_loading ** 2)
        error = rng.normal(0, math.sqrt(error_var * noise * noise), n)
        
        # 连续分数转换为1-5量表
        continuous_score = mean + true_score + error
        
        # 关键改进5: 使用更平滑的转换
        percentiles = np.percentile(continuous_score, [10, 30, 50, 70, 90])
        items[:, i] = np.digitize(continuous_score, percentiles) + 1
        items[:, i] = np.clip(items[:, i], 1, 5)
    
    return items.astype(int)


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
                    "use_Q1": True, "use_Q2": True, "use_Q3": True, "use_Q4": True, "use_Q5": True,
                    "Q1_perc": [50.0, 50.0],
                    "grade_levels": 3,
                    "Q2_perc": [35.0, 40.0, 25.0],
                    "Q3_perc": [55.0, 45.0],
                    "Q4_perc": [28.0, 72.0],
                    "Q5_perc": [38.0, 62.0],
                },
                "item_params": {"mean": 3.5, "loading": 0.85, "noise": 0.38},  # 改进的默认参数
                "corr_matrix": None,
                "mediation": {
                    "A": "A_dim", "C": "A_dim", "B": "A_dim",
                    "a": 0.6, "b": 0.6, "cprime": 0.2, "type": "部分中介",
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

        # --- 小维度设置 ---
        st.markdown("### 小维度设置（可选，大维度内部再分）")
        st.caption("每个大维度下可以再划分若干小维度。每行格式：小维度名称:题号,题号,题号")

        subdims_all = cfg.get("subdimensions", {})
        if not isinstance(subdims_all, dict):
            subdims_all = {}
        new_subdims_all = {}

        for big_dim, qids_big in cfg["dimensions"].items():
            with st.expander(f"大维度「{big_dim}」的小维度设置", expanded=False):
                exist_sub = subdims_all.get(big_dim, {})
                lines = []
                for sub_name, qids in exist_sub.items():
                    if qids:
                        lines.append(f"{sub_name}:{','.join(str(q) for q in qids)}")
                txt = st.text_area(
                    "每行一个小维度（示例：行为拖延:6,7,8,9,10）",
                    value="\n".join(lines),
                    key=f"subdim_text_{big_dim}",
                    height=120,
                )
                sub_dict = {}
                for line in txt.splitlines():
                    line = line.strip().replace("：", ":").replace("，", ",")
                    if not line or ":" not in line:
                        continue
                    name_part, q_part = line.split(":", 1)
                    sub_name = name_part.strip()
                    if not sub_name:
                        continue
                    qid_list_tmp = []
                    for token in q_part.split(","):
                        token = token.strip()
                        if token.isdigit():
                            q_val = int(token)
                            if q_val in qids_big:
                                qid_list_tmp.append(q_val)
                    if qid_list_tmp:
                        sub_dict[sub_name] = sorted(set(qid_list_tmp))
                new_subdims_all[big_dim] = sub_dict
                if sub_dict:
                    st.info("当前小维度配置：\n" + "\n".join([f"{k}: {v}" for k, v in sub_dict.items()]))
                else:
                    st.info("当前未设置小维度。")
        cfg["subdimensions"] = new_subdims_all

        # --- 人口学变量设置 ---
        st.markdown("### 人口学变量设置")
        demo = cfg.get("demo", {})
        st.markdown("**启用以下人口学变量：**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            demo["use_Q1"] = st.checkbox("性别", value=demo.get("use_Q1", True))
        with col2:
            demo["use_Q2"] = st.checkbox("年级", value=demo.get("use_Q2", True))
        with col3:
            demo["use_Q3"] = st.checkbox("生源地", value=demo.get("use_Q3", True))
        with col4:
            demo["use_Q4"] = st.checkbox("班干部", value=demo.get("use_Q4", True))
        with col5:
            demo["use_Q5"] = st.checkbox("独生子女", value=demo.get("use_Q5", True))

        st.markdown("**百分比设置：**")

        if demo.get("use_Q1", True):
            st.markdown("**性别比例（男/女）**")
            col1, col2 = st.columns(2)
            with col1:
                male_perc = st.number_input("男性比例(%)", 0.0, 100.0,
                                            float(demo.get("Q1_perc", [50.0, 50.0])[0]), 1.0, key="male_perc")
            with col2:
                st.metric("女性比例(%)", f"{100.0 - male_perc:.1f}")
            demo["Q1_perc"] = [male_perc, 100.0 - male_perc]

        if demo.get("use_Q2", True):
            st.markdown("**年级分布**")
            grade_levels = st.selectbox("年级数量", [3, 4],
                                        index=0 if demo.get("grade_levels", 3) == 3 else 1,
                                        key="grade_levels")
            demo["grade_levels"] = grade_levels
            if grade_levels == 3:
                default_perc = demo.get("Q2_perc", [35.0, 40.0, 25.0])
                col1, col2, col3 = st.columns(3)
                with col1:
                    perc1 = st.number_input("大一比例(%)", 0.0, 100.0, float(default_perc[0]), 1.0, key="grade1")
                with col2:
                    perc2 = st.number_input("大二比例(%)", 0.0, 100.0, float(default_perc[1]), 1.0, key="grade2")
                with col3:
                    st.metric("大三比例(%)", f"{100.0 - perc1 - perc2:.1f}")
                demo["Q2_perc"] = [perc1, perc2, 100.0 - perc1 - perc2]
            else:
                default_perc = demo.get("Q2_perc", [25.0, 30.0, 25.0, 20.0])
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    perc1 = st.number_input("大一比例(%)", 0.0, 100.0, float(default_perc[0]), 1.0, key="grade1_4")
                with col2:
                    perc2 = st.number_input("大二比例(%)", 0.0, 100.0, float(default_perc[1]), 1.0, key="grade2_4")
                with col3:
                    perc3 = st.number_input("大三比例(%)", 0.0, 100.0, float(default_perc[2]), 1.0, key="grade3_4")
                with col4:
                    st.metric("大四比例(%)", f"{100.0 - perc1 - perc2 - perc3:.1f}")
                demo["Q2_perc"] = [perc1, perc2, perc3, 100.0 - perc1 - perc2 - perc3]

        if demo.get("use_Q3", True):
            st.markdown("**生源地比例（城镇/农村）**")
            col1, col2 = st.columns(2)
            with col1:
                urban_perc = st.number_input("城镇比例(%)", 0.0, 100.0,
                                             float(demo.get("Q3_perc", [55.0, 45.0])[0]), 1.0, key="urban_perc")
            with col2:
                st.metric("农村比例(%)", f"{100.0 - urban_perc:.1f}")
            demo["Q3_perc"] = [urban_perc, 100.0 - urban_perc]

        if demo.get("use_Q4", True):
            st.markdown("**班干部比例（是/否）**")
            col1, col2 = st.columns(2)
            with col1:
                cadre_perc = st.number_input("班干部比例(%)", 0.0, 100.0,
                                             float(demo.get("Q4_perc", [28.0, 72.0])[0]), 1.0, key="cadre_perc")
            with col2:
                st.metric("非班干部比例(%)", f"{100.0 - cadre_perc:.1f}")
            demo["Q4_perc"] = [cadre_perc, 100.0 - cadre_perc]

        if demo.get("use_Q5", True):
            st.markdown("**独生子女比例（独生/非独生）**")
            col1, col2 = st.columns(2)
            with col1:
                only_perc = st.number_input("独生子女比例(%)", 0.0, 100.0,
                                            float(demo.get("Q5_perc", [38.0, 62.0])[0]), 1.0, key="only_perc")
            with col2:
                st.metric("非独生子女比例(%)", f"{100.0 - only_perc:.1f}")
            demo["Q5_perc"] = [only_perc, 100.0 - only_perc]

        cfg["demo"] = demo

        # --- 题目参数设置（改进的默认值） ---
        st.markdown("### 题目参数设置")
        st.caption("💡 提示：载荷越高、噪声越低，可靠性和相关性越好")
        item_params = cfg.get("item_params", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            item_params["mean"] = st.number_input("题目均值", 1.0, 5.0, float(item_params.get("mean", 3.6)), 0.1)
        with col2:
            item_params["loading"] = st.number_input("因子载荷", 0.5, 0.95, float(item_params.get("loading", 0.85)), 0.05)
        with col3:
            item_params["noise"] = st.number_input("噪声水平", 0.1, 1.0, float(item_params.get("noise", 0.45)), 0.05)
        cfg["item_params"] = item_params

        rev_txt = st.text_input(
            "反向题题号（逗号分隔）",
            value=",".join(map(str, cfg.get("reverse_items", []))) if cfg.get("reverse_items") else "",
            help="例如：8,12,15 表示这些题目在 1~5 上按 1↔5 反向计分。",
        )
        cfg["reverse_items"] = [int(x.strip()) for x in rev_txt.split(",") if x.strip().isdigit()]

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
                        for f in ["use_Q1", "use_Q2", "use_Q3", "use_Q4", "use_Q5"]:
                            demo_obj.setdefault(f, True)
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
            M = (M + M.T) / 2.0
            M = np.clip(M, -0.85, 0.85)  # 稍微放宽限制
            cfg["corr_matrix"] = M.tolist()

            st.markdown("### 中介：A→C→B（路径可正可负）")
            st.caption("💡 建议：路径系数在 ±0.3 到 ±0.8 之间效果较好")
            med = cfg.get("mediation", {})
            A_default = med.get("A", dims[0])
            C_default = med.get("C", dims[min(1, len(dims) - 1)])
            B_default = med.get("B", dims[min(2, len(dims) - 1)])
            if A_default not in dims: A_default = dims[0]
            if C_default not in dims: C_default = dims[min(1, len(dims) - 1)]
            if B_default not in dims: B_default = dims[min(2, len(dims) - 1)]

            A = st.selectbox("A（自变量）", dims, index=dims.index(A_default), key="med_A")
            C = st.selectbox("C（中介变量）", dims, index=dims.index(C_default), key="med_C")
            B = st.selectbox("B（因变量）", dims, index=dims.index(B_default), key="med_B")
            med_type = st.radio(
                "中介类型", ["完全中介", "部分中介"],
                index=1 if med.get("type", "部分中介") == "部分中介" else 0, horizontal=True,
            )
            a_default = float(med.get("a", 0.6))
            b_default = float(med.get("b", 0.6))
            cprime_default = float(med.get("cprime", 0.3 if med_type == "部分中介" else 0.0))

            a = st.slider("路径 a（A→C，可正可负）", -1.0, 1.0, a_default, 0.05)
            b_path = st.slider("路径 b（C→B，可正可负）", -1.0, 1.0, b_default, 0.05)
            cprime = 0.0
            if med_type == "部分中介":
                cprime = st.slider("直接效应 c'（A→B，可正可负）", -0.8, 0.8, cprime_default, 0.05)

            cfg["mediation"] = {
                "A": A, "C": C, "B": B,
                "a": float(a), "b": float(b_path), "cprime": float(cprime), "type": med_type,
            }
            st.session_state.config = cfg
            st.success("已保存关系约束。")


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
            N = int(cfg.get("N", 630))
            seed = int(cfg.get("seed", 42))

            all_qids = sorted([q["qid"] for q in qs])
            min_qid = min(all_qids)
            max_qid = max(all_qids)
            scale_start_qid = int(cfg.get("scale_start_qid", 6))
            scale_start_qid = max(min_qid, min(max_qid, scale_start_qid))

            k = len(dim_names)
            R = np.eye(k)
            cm = cfg.get("corr_matrix")
            if (isinstance(cm, list) and len(cm) == k
                    and all(isinstance(r, list) and len(r) == k for r in cm)):
                try:
                    R = np.array(cm, dtype=float)
                except Exception:
                    R = np.eye(k)

            if st.button("生成数据", type="primary"):
                rng = np.random.default_rng(seed)
                demo = cfg.get("demo", {})

                # 1) 人口学变量
                q1_perc = demo.get("Q1_perc", [50.0, 50.0])
                probs_gender = np.array(q1_perc[:2], dtype=float)
                probs_gender /= probs_gender.sum()
                gender_cat = rng.choice([1, 2], size=N, p=probs_gender)
                gender01 = (gender_cat == 2).astype(float)

                grade_levels = int(demo.get("grade_levels", 3))
                if grade_levels not in (3, 4):
                    grade_levels = 3
                q2_perc = demo.get("Q2_perc", [35.0, 40.0, 25.0, 0.0])
                probs_q2 = np.array(q2_perc[:grade_levels], dtype=float)
                probs_q2 /= probs_q2.sum()
                grade_cat = rng.choice(list(range(1, grade_levels + 1)), size=N, p=probs_q2)
                grade_num = (grade_cat - 1).astype(float) / (grade_levels - 1)  # 标准化到[0,1]

                q3p = demo.get("Q3_perc", [55.0, 45.0])
                probs_q3 = np.array(q3p[:2], dtype=float)
                probs_q3 /= probs_q3.sum()
                origin_cat = rng.choice([1, 2], size=N, p=probs_q3)
                origin01 = (origin_cat == 2).astype(float)

                q4p = demo.get("Q4_perc", [28.0, 72.0])
                probs_q4 = np.array(q4p[:2], dtype=float)
                probs_q4 /= probs_q4.sum()
                cadre_cat = rng.choice([1, 2], size=N, p=probs_q4)
                cadre01 = (cadre_cat == 1).astype(float)

                q5p = demo.get("Q5_perc", [38.0, 62.0])
                probs_q5 = np.array(q5p[:2], dtype=float)
                probs_q5 /= probs_q5.sum()
                only_cat = rng.choice([1, 2], size=N, p=probs_q5)
                only01 = (only_cat == 1).astype(float)

                # 2) 潜变量（改进的生成）
                Z = generate_latents(N, dim_names, corr_matrix=R, seed=seed)

                # 3) 中介结构（改进的应用）
                med = cfg.get("mediation")
                if med:
                    A_med = med.get("A"); C_med = med.get("C"); B_med = med.get("B")
                    if (A_med in Z.columns and C_med in Z.columns and B_med in Z.columns):
                        Z = apply_mediation(
                            Z, A_med, C_med, B_med,
                            a=float(med.get("a", 0.6)),
                            b=float(med.get("b", 0.6)),
                            cprime=float(med.get("cprime", 0.1)),
                            seed=seed + 7,
                        )

                # 4) 人口学差异 β（改进的应用方式）
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
                        # 标准化人口学变量的影响
                        delta = (b_gender * (gender01 - 0.5) * 2 +  # 转换为[-1,1]
                                b_grade * (grade_num - 0.5) * 2 +
                                b_origin * (origin01 - 0.5) * 2 +
                                b_cadre * (cadre01 - 0.5) * 2 +
                                b_only * (only01 - 0.5) * 2)
                        Z[d] = Z[d] + delta

                # 5) 输出 DataFrame
                out = pd.DataFrame({"ID": np.arange(1, N + 1)})
                if demo.get("use_Q1", False):
                    out["Q1"] = gender_cat
                if demo.get("use_Q2", False):
                    out["Q2"] = grade_cat
                if demo.get("use_Q3", False):
                    out["Q3"] = origin_cat
                if demo.get("use_Q4", False):
                    out["Q4"] = cadre_cat
                if demo.get("use_Q5", False):
                    out["Q5"] = only_cat

                # 6) 潜变量 → 题目分数（改进的生成）
                qid_to_dim = {}
                for d in dim_names:
                    for qid in dims_map[d]:
                        qid_to_dim[qid] = d
                rev = set(cfg.get("reverse_items", []))
                item_params = cfg.get("item_params", {})
                item_mean = float(item_params.get("mean", 3.6))
                item_loading = float(item_params.get("loading", 0.85))
                item_noise = float(item_params.get("noise", 0.45))

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
                            x = 6 - x
                        out[f"Q{qid}"] = x

                # 7) 大维度均分
                cols = ["ID"]
                for qid in all_qids:
                    if f"Q{qid}" in out.columns:
                        cols.append(f"Q{qid}")
                for d in dim_names:
                    qcols = [f"Q{qid}" for qid in dims_map[d] if f"Q{qid}" in out.columns]
                    if qcols:
                        out[f"{d}_mean"] = out[qcols].mean(axis=1)
                        cols.append(f"{d}_mean")

                # 8) 小维度均分（改进的Gram-Schmidt正交噪声）
                subdims_all = cfg.get("subdimensions", {})
                med_cfg = cfg.get("mediation") or {}
                med_A_name = med_cfg.get("A")
                med_C_name = med_cfg.get("C")
                med_B_name = med_cfg.get("B")
                a_val = float(med_cfg.get("a", 0.6)) if med_cfg else 0.0
                b_val = float(med_cfg.get("b", 0.6)) if med_cfg else 0.0

                def _make_target_unit(arr):
                    arr = arr.astype(float)
                    s = arr.std()
                    if s < 1e-8:
                        return None
                    return (arr - arr.mean()) / s

                if isinstance(subdims_all, dict):
                    for big_dim, subdict in subdims_all.items():
                        if not isinstance(subdict, dict):
                            continue
                        sub_names = [k for k in subdict.keys() if subdict[k]]
                        if not sub_names:
                            continue

                        # 确定目标向量
                        target_unit = None
                        if big_dim == med_A_name:
                            vecs = []
                            if f"{med_A_name}_mean" in out.columns:
                                v = _make_target_unit(out[f"{med_A_name}_mean"].to_numpy())
                                if v is not None:
                                    vecs.append(v)
                            if med_C_name and f"{med_C_name}_mean" in out.columns:
                                v = _make_target_unit(out[f"{med_C_name}_mean"].to_numpy())
                                if v is not None:
                                    sign = float(np.sign(a_val)) if a_val != 0 else 1.0
                                    vecs.append(v * sign)
                            if vecs:
                                combined = sum(vecs) / len(vecs)
                                target_unit = _make_target_unit(combined)

                        elif big_dim == med_C_name:
                            if med_B_name and f"{med_B_name}_mean" in out.columns:
                                v = _make_target_unit(out[f"{med_B_name}_mean"].to_numpy())
                                if v is not None:
                                    sign = float(np.sign(b_val)) if b_val != 0 else 1.0
                                    target_unit = v * sign

                        # Gram-Schmidt正交基（改进的相关性控制）
                        r_desired = 0.35  # 稍微提高目标相关性
                        basis = []
                        if target_unit is not None:
                            basis = [target_unit]

                        for si, sub_name in enumerate(sub_names):
                            qcols = [f"Q{qid}" for qid in subdict[sub_name]
                                     if f"Q{qid}" in out.columns]
                            if not qcols:
                                continue
                            safe_sub = re.sub(r"\W+", "", sub_name)
                            col_name = f"{big_dim}_{safe_sub}_mean"
                            raw_mean = out[qcols].mean(axis=1).to_numpy().astype(float)
                            raw_mu, raw_std = raw_mean.mean(), raw_mean.std()

                            if target_unit is not None:
                                # 生成与所有已有基正交的噪声
                                rng_sub = np.random.default_rng(seed + si + abs(hash(sub_name)) % 99999)
                                noise = rng_sub.standard_normal(N)
                                for b in basis:
                                    noise = noise - np.dot(noise, b) / (np.dot(b, b) + 1e-8) * b
                                noise_std = noise.std()
                                if noise_std < 1e-8:
                                    noise = rng_sub.standard_normal(N)
                                    noise_std = noise.std()
                                noise = (noise - noise.mean()) / noise_std
                                basis.append(noise)  # 加入正交基

                                sub_z = r_desired * target_unit + math.sqrt(max(1 - r_desired**2, 1e-8)) * noise
                                if raw_std > 1e-8:
                                    sub_final = sub_z * raw_std + raw_mu
                                else:
                                    sub_final = sub_z + raw_mu
                            else:
                                sub_final = raw_mean.copy()

                            out[col_name] = sub_final
                            cols.append(col_name)

                out = out[cols]
                st.session_state.generated = out
                st.success(f"✅ 已生成 {N} 行 × {out.shape[1]} 列数据，采用改进算法提升统计特性。")
                                # 实时可靠性检查
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
                grade_levels_vl = int(demo.get("grade_levels", 3))
                if grade_levels_vl not in (3, 4):
                    grade_levels_vl = 3
                if "Q1" in out.columns:
                    value_labels["Q1"] = {1: "男", 2: "女"}
                if "Q2" in out.columns:
                    value_labels["Q2"] = ({1: "大一", 2: "大二", 3: "大三"}
                                          if grade_levels_vl == 3
                                          else {1: "大一", 2: "大二", 3: "大三", 4: "大四"})
                if "Q3" in out.columns:
                    value_labels["Q3"] = {1: "城镇", 2: "农村"}
                if "Q4" in out.columns:
                    value_labels["Q4"] = {1: "是", 2: "否"}
                if "Q5" in out.columns:
                    value_labels["Q5"] = {1: "独生子女", 2: "非独生子女"}
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
                cfg["item_params"]["loading"] = 0.85
                cfg["item_params"]["noise"] = 0.38
                cfg["item_params"]["mean"] = 3.5
                st.session_state.config = cfg
                st.success("✅ 已应用高可靠性参数，请返回第4页重新生成数据")
