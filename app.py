
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="StockSense", page_icon="📦", layout="wide")

DEFAULT_LEAD_TIME = 4
DEFAULT_SAFETY_STOCK = 2


REQUIRED_COLS = [
    "date", "store", "item", "item_name", "prev_inventory",
    "stockout_risk", "sales", "Predicted_Demand", "reorder_qt", "action_needed",
]


def run_data_quality_checks(df: pd.DataFrame):
    """Clean common real-world data problems and report what was found.

    Returns (clean_df, issues) where issues is a list of (severity, message)
    tuples. severity is "error" (blocks the row from being used) or
    "warning" (informational — the app still runs).
    """
    issues = []
    df = df.copy()


    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        issues.append(("error", f"Missing required column(s): {', '.join(missing_cols)}."))
        return df, issues

    n0 = len(df)


    exact_dupes = df.duplicated().sum()
    if exact_dupes:
        df = df.drop_duplicates()
        issues.append(("warning", f"Removed {exact_dupes:,} exact duplicate row(s)."))


    key_dupes = df.duplicated(subset=["store", "item", "date"]).sum()
    if key_dupes:
        df = df.sort_values("date").drop_duplicates(subset=["store", "item", "date"], keep="last")
        issues.append((
            "warning",
            f"Found {key_dupes:,} row(s) with the same store/item/date but conflicting "
            f"values — kept the most recently listed entry for each and dropped the rest."
        ))


    null_mask = df[REQUIRED_COLS].isnull().any(axis=1)
    n_null = int(null_mask.sum())
    if n_null:
        df = df[~null_mask]
        issues.append(("warning", f"Dropped {n_null:,} row(s) with missing values in required fields."))


    bad_mask = (df["prev_inventory"] < 0) | (df["sales"] < 0) | (df["Predicted_Demand"] < 0)
    n_bad = int(bad_mask.sum())
    if n_bad:
        df = df[~bad_mask]
        issues.append(("warning", f"Dropped {n_bad:,} row(s) with negative stock, sales, or demand values."))

    if len(df) < n0:
        issues.append(("warning", f"{len(df):,} of {n0:,} rows kept after cleaning ({n0 - len(df):,} removed)."))


    if len(df):
        latest = df["date"].max()
        age_days = (pd.Timestamp.now().normalize() - latest.normalize()).days
        if age_days > 60:
            issues.append((
                "warning",
                f"Most recent record is {latest.date()} — {age_days:,} days old. "
                f"Recommendations may not reflect current conditions; refresh the pipeline output."
            ))

    return df, issues


@st.cache_data
def load_and_clean(file):
    raw = pd.read_csv(file)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    if "store" in raw.columns:
        raw["store"] = pd.to_numeric(raw["store"], errors="coerce")
    if "item" in raw.columns:
        raw["item"] = pd.to_numeric(raw["item"], errors="coerce")

    clean, issues = run_data_quality_checks(raw)

    blocking = [m for sev, m in issues if sev == "error"]
    if blocking:
        return None, issues

    clean["store"] = clean["store"].astype(int)
    clean["item"] = clean["item"].astype(int)
    clean = clean.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return clean, issues


st.sidebar.title("📦 StockSense")
st.sidebar.caption("AI Decision Engine · PS-04")

uploaded = st.sidebar.file_uploader("reorder_dataset.csv", type="csv")
st.sidebar.caption(
    "For re-testing with a refreshed pipeline output only. The file must already "
    "contain: date, store, item, item_name, prev_inventory, stockout_risk, sales, "
    "Predicted_Demand, reorder_qt, action_needed — this does not accept raw sales "
    "data or run the ML models."
)
data_path = uploaded if uploaded is not None else "reorder_dataset.csv"

try:
    data, quality_issues = load_and_clean(data_path)
    if data is None:
        st.title("📦 StockSense")
        for _, msg in quality_issues:
            st.error(msg)
        st.stop()
except FileNotFoundError:
    st.title("📦 StockSense")
    st.warning(
        "Couldn't find **reorder_dataset.csv** next to this app. "
        "Upload it using the file picker in the sidebar to get started."
    )
    st.stop()

if quality_issues:
    with st.expander(f"🧹 Data quality check — {len(quality_issues)} note(s) on this file", expanded=False):
        for _, msg in quality_issues:
            st.warning(msg)

st.sidebar.divider()


st.sidebar.subheader("What-if simulator")
lead_time = st.sidebar.slider("Lead time (days)", 1, 10, DEFAULT_LEAD_TIME)
safety_stock = st.sidebar.slider("Safety stock (days)", 0, 5, DEFAULT_SAFETY_STOCK)

st.sidebar.caption(
    "Reorder quantity = Predicted demand × (lead time + safety stock) − current stock"
)


coverage_days = lead_time + safety_stock
data["reorder_qt_live"] = np.maximum(
    (data["Predicted_Demand"] * coverage_days) - data["prev_inventory"], 0
).round().astype(int)
data["action_needed_live"] = np.where(
    (data["stockout_risk"] == 1) | (data["reorder_qt_live"] > 0), 1, 0
)


demand_safe = data["Predicted_Demand"].replace(0, np.nan)
data["days_of_stock"] = (data["prev_inventory"] / demand_safe).fillna(99).round(1)
data["priority"] = np.select(
    [
        (data["stockout_risk"] == 1) | (data["days_of_stock"] < lead_time),
        data["action_needed_live"] == 1,
    ],
    ["1 - Critical", "2 - High"],
    default="3 - OK",
)

st.sidebar.divider()
st.sidebar.subheader("Filters")
stores = sorted(data["store"].unique())
items = data[["item", "item_name"]].drop_duplicates().sort_values("item")
item_labels = {row.item: f"{row.item} · {row.item_name}" for row in items.itertuples()}

sel_store = st.sidebar.selectbox("Store", stores, format_func=lambda s: f"Store {s}")
sel_item = st.sidebar.selectbox("Product", items["item"], format_func=lambda i: item_labels[i])

min_date, max_date = data["date"].min(), data["date"].max()
sel_date = st.sidebar.slider(
    "As-of date",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=max_date.to_pydatetime(),
    format="YYYY-MM-DD",
)
sel_date = pd.Timestamp(sel_date)

today_all = data[data["date"] == sel_date].copy()
row = data[(data["store"] == sel_store) & (data["item"] == sel_item) & (data["date"] == sel_date)]


st.title("📦 StockSense")
st.caption(f"AI decision engine for retail inventory — snapshot for {sel_date.date()}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Products tracked", f"{data['item'].nunique()} × {data['store'].nunique()} stores")
k2.metric("Need action today", int(today_all["action_needed_live"].sum()))
k3.metric("At stockout risk", int(today_all["stockout_risk"].sum()))
k4.metric(
    "Total units to reorder",
    f"{int(today_all['reorder_qt_live'].sum()):,}",
)

st.divider()


left, right = st.columns([2, 1])

with left:
    st.subheader(f"{item_labels[sel_item]} — Store {sel_store}")

    hist = data[(data["store"] == sel_store) & (data["item"] == sel_item)].sort_values("date")
    window = hist[(hist["date"] >= sel_date - pd.Timedelta(days=45)) & (hist["date"] <= sel_date)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=window["date"], y=window["prev_inventory"], name="Stock on hand",
        line=dict(color="#2E86AB", width=2), fill="tozeroy", fillcolor="rgba(46,134,171,0.1)",
    ))
    reorder_point = window["Predicted_Demand"].mean() * (lead_time + safety_stock)
    fig.add_hline(
        y=reorder_point, line_dash="dash", line_color="#E67E22",
        annotation_text="Reorder point", annotation_position="top left",
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Units in stock", xaxis_title=None,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=window["date"], y=window["sales"], name="Actual sales", marker_color="#95A5A6"))
    fig2.add_trace(go.Scatter(
        x=window["date"], y=window["Predicted_Demand"], name="Predicted demand",
        line=dict(color="#C0392B", width=2),
    ))
    fig2.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Units", legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("Recommendation")
    if row.empty:
        st.info("No record for this store/product on the selected date.")
    else:
        r = row.iloc[0]
        stock = r["prev_inventory"]
        demand = r["Predicted_Demand"]
        reorder_qty = int(np.maximum((demand * coverage_days) - stock, 0).round())
        days_covered = stock / demand if demand > 0 else float("inf")

        if r["stockout_risk"] == 1:
            st.error("⚠️ Stockout risk today")
        elif reorder_qty > 0:
            st.warning("🔁 Reorder recommended")
        else:
            st.success("✅ Stock healthy")

        st.metric("Recommended reorder", f"{reorder_qty} units")
        st.metric("Current stock", f"{stock:.0f} units")
        st.metric("Predicted demand", f"{demand:.0f} units/day")

        st.caption(
            f"Current stock ({stock:.0f}) covers about **{days_covered:.1f} days** "
            f"of expected demand ({demand:.0f}/day). Against a {lead_time}-day lead time "
            f"plus {safety_stock}-day safety buffer ({coverage_days} days total coverage), "
            f"we recommend ordering **{reorder_qty} units** now."
            if demand > 0 else
            "No recent demand recorded for this product — no reorder needed."
        )


if not row.empty:
    with st.expander("🔎 Trace this recommendation back to the data"):
        r = row.iloc[0]
        st.markdown("**Inputs used for this decision**")
        trace_inputs = pd.DataFrame({
            "Input": [
                "Current stock (start of day)",
                "Predicted demand (per day)",
                "Lead time (days)",
                "Safety stock (days)",
                "Coverage needed (lead + safety)",
                "Stockout classifier output",
            ],
            "Value": [
                f"{r['prev_inventory']:.1f} units",
                f"{r['Predicted_Demand']:.0f} units",
                lead_time,
                safety_stock,
                f"{coverage_days} days",
                "At risk" if r["stockout_risk"] == 1 else "Not at risk",
            ],
            "Comes from": [
                "Inventory records (stockout notebook)",
                "Demand forecasting model",
                "What-if slider",
                "What-if slider",
                "Lead time + safety stock",
                "Stockout classifier (XGBoost)",
            ],
        })
        st.dataframe(trace_inputs, use_container_width=True, hide_index=True)

        st.markdown(
            f"**Calculation:** {r['Predicted_Demand']:.0f} × {coverage_days} "
            f"− {r['prev_inventory']:.1f} = "
            f"{r['Predicted_Demand'] * coverage_days - r['prev_inventory']:.1f} "
            f"→ reorder **{int(r['reorder_qt_live'])}** units (never below 0)"
        )

        st.markdown("**Source rows: the last 7 days for this store and product**")
        last7 = (
            data[(data["store"] == sel_store) & (data["item"] == sel_item) & (data["date"] <= sel_date)]
            .sort_values("date")
            .tail(7)
            [["date", "prev_inventory", "sales", "Predicted_Demand", "stockout_risk"]]
            .rename(columns={
                "date": "Date", "prev_inventory": "Stock", "sales": "Actual sales",
                "Predicted_Demand": "Predicted demand", "stockout_risk": "Stockout risk",
            })
        )
        last7["Date"] = last7["Date"].dt.date
        st.dataframe(last7, use_container_width=True, hide_index=True)

st.divider()


st.subheader(f"Needs attention today ({sel_date.date()})")

st.caption(
    "Ranked by urgency: items that will run out before a new order can arrive come first. "
    "The AI suggests a quantity — a manager can edit it, then approve or reject before anything is ordered."
)


action_list = (
    today_all[today_all["action_needed_live"] == 1]
    [["priority", "store", "item", "item_name", "prev_inventory", "days_of_stock",
      "Predicted_Demand", "stockout_risk", "reorder_qt_live"]]
    .sort_values(["priority", "days_of_stock", "reorder_qt_live"], ascending=[True, True, False])
    .rename(columns={
        "priority": "Priority", "item_name": "Product", "store": "Store", "item": "Item ID",
        "prev_inventory": "Stock", "days_of_stock": "Days of stock",
        "Predicted_Demand": "Predicted demand",
        "stockout_risk": "Stockout risk", "reorder_qt_live": "Suggested qty",
    })
    .reset_index(drop=True)
)
action_list["Priority"] = action_list["Priority"].str[4:]
action_list["Stockout risk"] = action_list["Stockout risk"].map({1: "⚠️ Yes", 0: ""})
action_list.insert(0, "Decision", "Pending")

action_list.insert(len(action_list.columns), "Reorder qty", action_list["Suggested qty"])

n_critical = int((action_list["Priority"] == "Critical").sum())
c1, c2 = st.columns(2)
c1.metric("Critical (out before the next delivery)", n_critical)
c2.metric("High priority", len(action_list) - n_critical)


editor_key = f"approvals_{sel_date.date()}_{lead_time}_{safety_stock}"
edited = st.data_editor(
    action_list,
    key=editor_key,
    use_container_width=True,
    hide_index=True,
    height=320,
    disabled=[c for c in action_list.columns if c not in ("Decision", "Reorder qty")],
    column_config={
        "Decision": st.column_config.SelectboxColumn(
            "Decision", options=["Pending", "Approve", "Reject"],
            help="Approve to order, reject to dismiss — both are logged.",
        ),
        "Reorder qty": st.column_config.NumberColumn(
            "Reorder qty", min_value=0, step=1,
            help="Editable — defaults to the AI's suggested quantity.",
        ),
    },
)

approved = edited[edited["Decision"] == "Approve"].drop(columns=["Decision"])
rejected = edited[edited["Decision"] == "Reject"].drop(columns=["Decision"])
edited_qty = int((edited["Reorder qty"] != edited["Suggested qty"]).sum())

st.markdown("**Manager decision**")
a1, a2, a3 = st.columns([1, 1, 1])
approver = a1.text_input("Your name", placeholder="Required to submit")
a2.metric("Approved", f"{len(approved)} · {int(approved['Reorder qty'].sum()):,} units")
a3.metric("Rejected", len(rejected))
if edited_qty:
    st.caption(f"✏️ {edited_qty} quantity override(s) made vs. the AI's suggestion.")

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if len(approved) == 0 and len(rejected) == 0:
    st.info("Set a Decision (Approve/Reject) on the rows below, then submit.")
elif not approver.strip():
    st.warning("Enter your name above to submit these decisions.")
else:
    if st.button("✅ Submit decisions to the audit log"):
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        for _, r in approved.iterrows():
            override = r["Reorder qty"] != r["Suggested qty"]
            st.session_state.audit_log.append({
                "Timestamp": now, "Decision": "Approved", "By": approver.strip(),
                "Store": r["Store"], "Item ID": r["Item ID"], "Product": r["Product"],
                "Suggested qty": r["Suggested qty"], "Final qty": r["Reorder qty"],
                "Overridden": "Yes" if override else "No",
            })
        for _, r in rejected.iterrows():
            st.session_state.audit_log.append({
                "Timestamp": now, "Decision": "Rejected", "By": approver.strip(),
                "Store": r["Store"], "Item ID": r["Item ID"], "Product": r["Product"],
                "Suggested qty": r["Suggested qty"], "Final qty": r["Reorder qty"],
                "Overridden": "N/A",
            })
        st.success(f"Logged {len(approved)} approval(s) and {len(rejected)} rejection(s).")

    if len(approved):
        export = approved.copy()
        export["Approved by"] = approver.strip()
        export["Approved at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        st.download_button(
            "⬇️ Export approved orders (CSV)",
            export.to_csv(index=False).encode("utf-8"),
            f"approved_orders_{sel_date.date()}.csv",
            "text/csv",
        )

with st.expander(f"📜 Audit log — {len(st.session_state.get('audit_log', []))} decision(s) this session"):
    if st.session_state.get("audit_log"):
        log_df = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download full audit log (CSV)",
            log_df.to_csv(index=False).encode("utf-8"),
            "audit_log.csv", "text/csv",
        )
        if st.button("Clear audit log"):
            st.session_state.audit_log = []
            st.rerun()
    else:
        st.caption("No decisions submitted yet this session.")

with st.expander("Download the full ranked list (unapproved)"):
    st.download_button(
        "⬇️ Download ranked list (CSV)",
        action_list.drop(columns=["Decision"]).to_csv(index=False).encode("utf-8"),
        f"reorder_list_{sel_date.date()}.csv",
        "text/csv",
    )

st.divider()


st.subheader("Risk heatmap — all stores × products")

pivot = today_all.pivot_table(
    index="item", columns="store", values="action_needed_live", aggfunc="max"
).fillna(0)
pivot.index = [item_labels[i].split(" · ")[1] for i in pivot.index]

fig3 = px.imshow(
    pivot, aspect="auto", color_continuous_scale=["#E8F4F8", "#C0392B"],
    labels=dict(x="Store", y="Product", color="Needs action"),
)
fig3.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
fig3.update_xaxes(side="top")
st.plotly_chart(fig3, use_container_width=True)
