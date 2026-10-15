import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.min_cut import exact_balanced_min_cut, kernighan_lin, validate_graph, EXACT_SEARCH_LIMIT
from algorithms.lee_routing import lee_route_4dir, lee_route_8dir, validate_grid, random_obstacles, example_maze

st.set_page_config(page_title="VLSI CAD Simulator", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B1020;
    color: #F1F5F9;
}
.stApp { background-color: #0B1020; }

.stTabs [data-baseweb="tab-list"] {
    background: #121A2B;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #253552;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #9AA9BE;
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 500;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background: #8B7CFF !important;
    color: white !important;
    font-weight: 600;
}

.stButton > button {
    background: #182338;
    color: #F1F5F9;
    border: 1px solid #253552;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 14px;
}
.stButton > button:hover {
    background: #8B7CFF;
    border-color: #8B7CFF;
    color: white;
}

[data-testid="metric-container"] {
    background: #182338;
    border: 1px solid #253552;
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] { color: #9AA9BE !important; font-size: 11px !important; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 20px !important; font-weight: 700 !important; }

.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    background: #121A2B !important;
    border: 1px solid #253552 !important;
    border-radius: 6px !important;
    color: #F1F5F9 !important;
}
label { color: #9AA9BE !important; font-size: 12px !important; }
.stRadio label { color: #F1F5F9 !important; font-size: 13px !important; }
.stCheckbox label { color: #F1F5F9 !important; font-size: 13px !important; }

.streamlit-expanderHeader {
    background: #182338 !important;
    color: #F1F5F9 !important;
    border-radius: 6px !important;
    border: 1px solid #253552 !important;
}
.streamlit-expanderContent {
    background: #121A2B !important;
    border: 1px solid #253552 !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h2 style="color:#F1F5F9;margin-bottom:2px;font-size:22px;">VLSI CAD Lab</h2>', unsafe_allow_html=True)
st.markdown('<p style="color:#9AA9BE;margin-top:0;font-size:13px;">Interactive Placement & Routing Algorithm Explorer</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Min-Cut Partitioning", "Lee Maze Routing"])


# ─── helpers used in both tabs ───────────────────────────────────────────────

def default_graph():
    vertices = ["A", "B", "C", "D", "E", "F", "G", "H"]
    edges = [
        ("A","B"),("A","C"),("B","C"),("B","D"),
        ("C","D"),("C","E"),("D","E"),("D","F"),
        ("E","F"),("E","G"),("F","G"),("F","H"),
        ("G","H"),("A","D"),("B","E"),("G","D"),
    ]
    return vertices, edges


def load_default_grid():
    rows, cols, src, dst, obs = example_maze()
    st.session_state.r_rows = rows
    st.session_state.r_cols = cols
    st.session_state.r_src = src
    st.session_state.r_dst = dst
    st.session_state.r_obs = obs
    st.session_state.r_result = None


# ─── session state init ───────────────────────────────────────────────────────

if "g_vertices" not in st.session_state:
    v, e = default_graph()
    st.session_state.g_vertices = v
    st.session_state.g_edges = e
    st.session_state.g_result = None

if "r_rows" not in st.session_state:
    load_default_grid()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MIN-CUT PARTITIONING
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<p style="color:#9AA9BE;font-size:13px;margin-bottom:12px;">Divides the circuit graph into two balanced groups minimising the number of connections (cut edges) between them. For small graphs an exact search is used; for larger ones the Kernighan-Lin heuristic runs instead.</p>', unsafe_allow_html=True)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown('<p style="font-size:14px;font-weight:600;color:#F1F5F9;border-bottom:1px solid #253552;padding-bottom:6px;margin-bottom:12px;">Graph Editor</p>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Load Example", key="g_ex", width='stretch'):
                v, e = default_graph()
                st.session_state.g_vertices = v
                st.session_state.g_edges = e
                st.session_state.g_result = None
                st.rerun()
        with c2:
            if st.button("Random Graph", key="g_rand", width='stretch'):
                rng = random.Random(random.randint(0, 9999))
                n = rng.choice([6, 8, 10])
                verts = [chr(65+i) for i in range(n)]
                edges = [(verts[i], verts[j]) for i in range(n) for j in range(i+1, n) if rng.random() < 0.45]
                st.session_state.g_vertices = verts
                st.session_state.g_edges = edges
                st.session_state.g_result = None
                st.rerun()
        with c3:
            if st.button("Clear", key="g_clear", width='stretch'):
                st.session_state.g_vertices = []
                st.session_state.g_edges = []
                st.session_state.g_result = None
                st.rerun()

        with st.expander("Add / Remove Vertex"):
            vname = st.text_input("Vertex name", key="g_vname", placeholder="e.g. X")
            ca, cb = st.columns(2)
            with ca:
                if st.button("Add", key="g_addv", width='stretch'):
                    name = vname.strip().upper()
                    if not name:
                        st.error("Enter a name.")
                    elif name in st.session_state.g_vertices:
                        st.error("Already exists.")
                    else:
                        st.session_state.g_vertices.append(name)
                        st.session_state.g_result = None
                        st.rerun()
            with cb:
                if st.session_state.g_vertices:
                    dv = st.selectbox("Remove", ["--"] + st.session_state.g_vertices, key="g_dv", label_visibility="collapsed")
                    if st.button("Remove", key="g_remv", width='stretch') and dv != "--":
                        st.session_state.g_vertices.remove(dv)
                        st.session_state.g_edges = [e for e in st.session_state.g_edges if e[0] != dv and e[1] != dv]
                        st.session_state.g_result = None
                        st.rerun()

        with st.expander("Add / Remove Edge"):
            if len(st.session_state.g_vertices) < 2:
                st.info("Need at least 2 vertices.")
            else:
                eu = st.selectbox("From", st.session_state.g_vertices, key="g_eu")
                opts = [v for v in st.session_state.g_vertices if v != eu]
                ev = st.selectbox("To", opts, key="g_ev") if opts else None
                da, db = st.columns(2)
                with da:
                    if st.button("Add Edge", key="g_adde", width='stretch') and ev:
                        existing = [(e[0], e[1]) for e in st.session_state.g_edges]
                        if (eu, ev) in existing or (ev, eu) in existing:
                            st.error("Edge already exists.")
                        else:
                            st.session_state.g_edges.append((eu, ev))
                            st.session_state.g_result = None
                            st.rerun()
                with db:
                    labels = [f"{e[0]}-{e[1]}" for e in st.session_state.g_edges]
                    if labels:
                        de = st.selectbox("Edge to remove", ["--"] + labels, key="g_de", label_visibility="collapsed")
                        if st.button("Remove Edge", key="g_reme", width='stretch') and de != "--":
                            idx = labels.index(de)
                            st.session_state.g_edges.pop(idx)
                            st.session_state.g_result = None
                            st.rerun()

        st.markdown('<p style="font-size:14px;font-weight:600;color:#F1F5F9;border-bottom:1px solid #253552;padding-bottom:6px;margin:12px 0;">Settings</p>', unsafe_allow_html=True)

        n = len(st.session_state.g_vertices)
        if n > EXACT_SEARCH_LIMIT:
            st.info(f"Graph has {n} vertices — using Kernighan-Lin heuristic.")
            alg = "kl"
        else:
            alg = "exact" if st.radio("Method", ["Exact search", "Kernighan-Lin heuristic"], key="g_alg") == "Exact search" else "kl"

        balance = st.slider("Balance (fraction in group A)", 0.1, 0.9, 0.5, 0.05, key="g_bal")

        if st.button("Run Partitioning", key="g_run", width='stretch'):
            errs = validate_graph(st.session_state.g_vertices, st.session_state.g_edges)
            bad = [e for e in errs if any(k in e for k in ["at least 2", "unknown vertex", "non-positive"])]
            for e in bad:
                st.error(e)
            if not bad and len(st.session_state.g_vertices) >= 2:
                try:
                    if alg == "exact":
                        st.session_state.g_result = exact_balanced_min_cut(
                            st.session_state.g_vertices, st.session_state.g_edges,
                            balance_constraint=balance
                        )
                    else:
                        st.session_state.g_result = kernighan_lin(
                            st.session_state.g_vertices, st.session_state.g_edges
                        )
                    st.rerun()
                except ValueError as ex:
                    st.error(str(ex))

    with right:
        result = st.session_state.g_result
        verts = st.session_state.g_vertices
        edges = st.session_state.g_edges

        if verts:
            G = nx.Graph()
            G.add_nodes_from(verts)
            for e in edges:
                G.add_edge(e[0], e[1])
            pos = nx.spring_layout(G, seed=42, k=1.8 / max(1, len(verts)**0.5))

            g1 = set(result["group1"]) if result else set()
            g2 = set(result["group2"]) if result else set()
            cut_set = set()
            if result:
                for ce in result["cut_edges"]:
                    cut_set.add((min(ce["u"], ce["v"]), max(ce["u"], ce["v"])))

            edge_traces = []
            for u, v in edges:
                if u not in pos or v not in pos:
                    continue
                key = (min(u, v), max(u, v))
                color = "#FF6B81" if key in cut_set else "#253552"
                width = 2.5 if key in cut_set else 1.2
                x0, y0 = pos[u]; x1, y1 = pos[v]
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(color=color, width=width),
                    hoverinfo="none", showlegend=False,
                ))

            def node_color(v):
                if not result:
                    return "#54D6E8"
                if v in g1:
                    return "#8B7CFF"
                if v in g2:
                    return "#4DD4AC"
                return "#54D6E8"

            nx_list = [v for v in verts if v in pos]
            node_trace = go.Scatter(
                x=[pos[v][0] for v in nx_list],
                y=[pos[v][1] for v in nx_list],
                mode="markers+text",
                marker=dict(size=24, color=[node_color(v) for v in nx_list], line=dict(color="#0B1020", width=2)),
                text=nx_list,
                textposition="middle center",
                textfont=dict(size=11, color="white", family="Inter"),
                hoverinfo="text",
                showlegend=False,
            )

            fig = go.Figure(data=edge_traces + [node_trace])

            if result:
                fig.add_shape(type="line", x0=0, x1=0, y0=-1.6, y1=1.6,
                              line=dict(color="#253552", width=1, dash="dash"))
                fig.add_annotation(x=-0.7, y=1.5, text="Partition A",
                                   font=dict(color="#8B7CFF", size=11), showarrow=False)
                fig.add_annotation(x=0.7, y=1.5, text="Partition B",
                                   font=dict(color="#4DD4AC", size=11), showarrow=False)

            fig.update_layout(
                paper_bgcolor="#121A2B", plot_bgcolor="#121A2B",
                margin=dict(l=10, r=10, t=10, b=10), height=400,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        else:
            st.info("Add vertices using the editor on the left, then run partitioning.")

        if result:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cut Cost", f"{result['cut_cost']:.0f}")
            m2.metric("Balance", result["balance"])
            m3.metric("Runtime", f"{result['runtime_ms']:.2f} ms")
            m4.metric("Method", "Exact" if "Exact" in result["optimality"] else "Heuristic")

            st.markdown(f"""
            <div style="margin-top:8px;font-size:12px;color:#9AA9BE;">
            Group A: <b style="color:#8B7CFF">{", ".join(result["group1"])}</b>
            &nbsp;&nbsp;&nbsp;
            Group B: <b style="color:#4DD4AC">{", ".join(result["group2"])}</b>
            &nbsp;&nbsp;&nbsp;
            Cut edges: <b style="color:#FF6B81">{len(result["cut_edges"])}</b>
            &nbsp;&nbsp;&nbsp;
            {result["optimality"]}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:#9AA9BE;">
          <span><span style="display:inline-block;width:10px;height:10px;background:#54D6E8;border-radius:50%;margin-right:4px;"></span>Unassigned</span>
          <span><span style="display:inline-block;width:10px;height:10px;background:#8B7CFF;border-radius:50%;margin-right:4px;"></span>Partition A</span>
          <span><span style="display:inline-block;width:10px;height:10px;background:#4DD4AC;border-radius:50%;margin-right:4px;"></span>Partition B</span>
          <span><span style="display:inline-block;width:14px;height:2px;background:#FF6B81;margin-right:4px;vertical-align:middle;"></span>Cut edge</span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LEE MAZE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<p style="color:#9AA9BE;font-size:13px;margin-bottom:12px;">Lee\'s algorithm uses BFS wave expansion to find the shortest path from source to destination on a grid, respecting obstacles. 4-directional movement uses equal unit cost; 8-directional uses Dijkstra with diagonal cost = sqrt(2).</p>', unsafe_allow_html=True)

    rleft, rright = st.columns([1, 2.5], gap="large")

    with rleft:
        st.markdown('<p style="font-size:14px;font-weight:600;color:#F1F5F9;border-bottom:1px solid #253552;padding-bottom:6px;margin-bottom:12px;">Grid Controls</p>', unsafe_allow_html=True)

        qa, qb, qc = st.columns(3)
        with qa:
            if st.button("Example Maze", key="r_ex", width='stretch'):
                load_default_grid()
                st.rerun()
        with qb:
            if st.button("Random Obs.", key="r_rand", width='stretch'):
                st.session_state.r_obs = random_obstacles(
                    st.session_state.r_rows, st.session_state.r_cols,
                    st.session_state.r_src, st.session_state.r_dst,
                    density=0.25, seed=random.randint(0, 9999)
                )
                st.session_state.r_result = None
                st.rerun()
        with qc:
            if st.button("Clear Obs.", key="r_clear", width='stretch'):
                st.session_state.r_obs = set()
                st.session_state.r_result = None
                st.rerun()

        with st.expander("Grid Size", expanded=True):
            nr = st.number_input("Rows", 3, 50, st.session_state.r_rows, 1, key="r_nr")
            nc = st.number_input("Columns", 3, 80, st.session_state.r_cols, 1, key="r_nc")
            if st.button("Apply Size", key="r_appsz", width='stretch'):
                st.session_state.r_rows = int(nr)
                st.session_state.r_cols = int(nc)
                st.session_state.r_obs = {(r,c) for r,c in st.session_state.r_obs if r < nr and c < nc}
                sr, sc2 = st.session_state.r_src
                dr2, dc2 = st.session_state.r_dst
                if not (sr < nr and sc2 < nc):
                    st.session_state.r_src = (0, 0)
                    st.warning("Source reset to (0,0) after resize.")
                if not (dr2 < nr and dc2 < nc):
                    st.session_state.r_dst = (int(nr)-1, int(nc)-1)
                    st.warning("Destination reset to bottom-right after resize.")
                st.session_state.r_result = None
                st.rerun()

        with st.expander("Source & Destination", expanded=True):
            st.caption("Row, column — 0-indexed from top-left")
            s1, s2 = st.columns(2)
            with s1:
                srow = st.number_input("Source row", 0, st.session_state.r_rows-1, st.session_state.r_src[0], key="r_srow")
                scol = st.number_input("Source col", 0, st.session_state.r_cols-1, st.session_state.r_src[1], key="r_scol")
            with s2:
                drow = st.number_input("Dest row", 0, st.session_state.r_rows-1, st.session_state.r_dst[0], key="r_drow")
                dcol = st.number_input("Dest col", 0, st.session_state.r_cols-1, st.session_state.r_dst[1], key="r_dcol")
            if st.button("Update S/D", key="r_upsd", width='stretch'):
                ns = (int(srow), int(scol))
                nd = (int(drow), int(dcol))
                if ns == nd:
                    st.error("Source and destination must differ.")
                elif ns in st.session_state.r_obs:
                    st.error("Source is blocked by an obstacle.")
                elif nd in st.session_state.r_obs:
                    st.error("Destination is blocked by an obstacle.")
                else:
                    st.session_state.r_src = ns
                    st.session_state.r_dst = nd
                    st.session_state.r_result = None
                    st.rerun()

        with st.expander("Obstacles"):
            ob1, ob2 = st.columns(2)
            with ob1:
                obr = st.number_input("Row", 0, st.session_state.r_rows-1, 0, key="r_obr")
            with ob2:
                obc = st.number_input("Col", 0, st.session_state.r_cols-1, 0, key="r_obc")
            oa, ob = st.columns(2)
            with oa:
                if st.button("Add", key="r_addob", width='stretch'):
                    cell = (int(obr), int(obc))
                    if cell == st.session_state.r_src:
                        st.error("Cannot block source.")
                    elif cell == st.session_state.r_dst:
                        st.error("Cannot block destination.")
                    else:
                        st.session_state.r_obs.add(cell)
                        st.session_state.r_result = None
                        st.rerun()
            with ob:
                if st.button("Remove", key="r_remob", width='stretch'):
                    st.session_state.r_obs.discard((int(obr), int(obc)))
                    st.session_state.r_result = None
                    st.rerun()

        st.markdown('<p style="font-size:14px;font-weight:600;color:#F1F5F9;border-bottom:1px solid #253552;padding-bottom:6px;margin:12px 0;">Algorithm</p>', unsafe_allow_html=True)

        move_mode = st.radio("Movement", ["4-directional (BFS, unit cost)", "8-directional (Dijkstra, diag=sqrt2)"], key="r_mode")

        if st.button("Run Routing", key="r_run", width='stretch'):
            errs = validate_grid(
                st.session_state.r_rows, st.session_state.r_cols,
                st.session_state.r_src, st.session_state.r_dst,
                st.session_state.r_obs
            )
            bad = [e for e in errs if any(k in e for k in ["must be", "outside", "blocked", "same cell"])]
            for e in bad:
                st.error(e)
            if not bad:
                if "4-directional" in move_mode:
                    st.session_state.r_result = lee_route_4dir(
                        st.session_state.r_rows, st.session_state.r_cols,
                        st.session_state.r_src, st.session_state.r_dst,
                        st.session_state.r_obs
                    )
                else:
                    st.session_state.r_result = lee_route_8dir(
                        st.session_state.r_rows, st.session_state.r_cols,
                        st.session_state.r_src, st.session_state.r_dst,
                        st.session_state.r_obs
                    )
                st.rerun()

    with rright:
        import numpy as np

        rows = st.session_state.r_rows
        cols = st.session_state.r_cols
        src = st.session_state.r_src
        dst = st.session_state.r_dst
        obs = st.session_state.r_obs
        res = st.session_state.r_result

        EMPTY, OBSTACLE, EXPLORED, PATH, SOURCE, DEST = 0, 1, 2, 3, 4, 5

        grid = np.zeros((rows, cols))
        hover = [[f"({r},{c})" for c in range(cols)] for r in range(rows)]

        for r, c in obs:
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = OBSTACLE

        if res:
            visited = set(tuple(x) for x in res.get("visited_order", []))
            path_set = set(tuple(x) for x in res.get("path", []))
            for r, c in visited:
                if 0 <= r < rows and 0 <= c < cols and grid[r][c] == EMPTY:
                    grid[r][c] = EXPLORED
            for r, c in path_set:
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r][c] = PATH

        sr, sc2 = src; dr2, dc2 = dst
        grid[sr][sc2] = SOURCE
        grid[dr2][dc2] = DEST

        text_labels = [["" for _ in range(cols)] for _ in range(rows)]
        text_labels[sr][sc2] = "S"
        text_labels[dr2][dc2] = "D"

        colorscale = [
            [0/5, "#1E3A5F"],   # empty — muted navy
            [1/5, "#4A5568"],   # obstacle — slate gray
            [2/5, "#2D4A6B"],   # explored — muted blue
            [3/5, "#FFC857"],   # path — amber
            [4/5, "#4DD4AC"],   # source — green
            [5/5, "#FF6B81"],   # destination — red
        ]

        cell_px = max(18, min(34, int(640 / max(cols, 1))))

        fig2 = go.Figure(go.Heatmap(
            z=grid,
            text=text_labels,
            texttemplate="%{text}",
            textfont=dict(size=max(8, cell_px-10), color="white", family="Inter"),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale=colorscale,
            zmin=0, zmax=5,
            showscale=False,
            xgap=1.5, ygap=1.5,
        ))

        fig2.update_layout(
            paper_bgcolor="#121A2B", plot_bgcolor="#121A2B",
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(300, rows * cell_px + 50),
            xaxis=dict(tickmode="linear", tick0=0, dtick=1,
                       tickfont=dict(size=9, color="#9AA9BE"), showgrid=False, zeroline=False),
            yaxis=dict(tickmode="linear", tick0=0, dtick=1,
                       tickfont=dict(size=9, color="#9AA9BE"), showgrid=False, zeroline=False,
                       autorange="reversed"),
        )

        st.markdown('<p style="font-size:14px;font-weight:600;color:#F1F5F9;border-bottom:1px solid #253552;padding-bottom:6px;margin-bottom:10px;">Routing Grid</p>', unsafe_allow_html=True)
        st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})

        st.markdown(f"""
        <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#9AA9BE;margin-top:4px;">
          <span><span style="display:inline-block;width:10px;height:10px;background:#1E3A5F;border:1px solid #253552;border-radius:2px;margin-right:4px;"></span>Empty</span>
          <span><span style="display:inline-block;width:10px;height:10px;background:#4A5568;border-radius:2px;margin-right:4px;"></span>Obstacle</span>
          <span><span style="display:inline-block;width:10px;height:10px;background:#4DD4AC;border-radius:2px;margin-right:4px;"></span>Source (S)</span>
          <span><span style="display:inline-block;width:10px;height:10px;background:#FF6B81;border-radius:2px;margin-right:4px;"></span>Destination (D)</span>
          <span><span style="display:inline-block;width:10px;height:10px;background:#2D4A6B;border-radius:2px;margin-right:4px;"></span>Explored</span>
          <span><span style="display:inline-block;width:10px;height:10px;background:#FFC857;border-radius:2px;margin-right:4px;"></span>Path</span>
        </div>
        """, unsafe_allow_html=True)

        if res:
            st.markdown('<p style="font-size:14px;font-weight:600;color:#F1F5F9;border-bottom:1px solid #253552;padding-bottom:6px;margin:14px 0 10px;">Results</p>', unsafe_allow_html=True)
            rm1, rm2, rm3, rm4 = st.columns(4)
            rm1.metric("Route", "Found" if res["found"] else "Not Found")
            rm2.metric("Path Length", f"{res['path_length']} moves" if res.get("path_length") is not None else "N/A")
            rm3.metric("Cells Explored", str(res.get("cells_explored", "-")))
            rm4.metric("Runtime", f"{res.get('runtime_ms', 0):.2f} ms")

            st.markdown(f"""
            <p style="font-size:12px;color:#9AA9BE;margin-top:6px;">
            Grid: {rows} x {cols} &nbsp;|&nbsp;
            Obstacles: {len(obs)} &nbsp;|&nbsp;
            Cells in path: {len(res.get("path", [])) if res["found"] else "N/A"} &nbsp;|&nbsp;
            Moves = cells - 1
            {f'&nbsp;|&nbsp; Total cost: {res["total_cost"]:.3f}' if res.get("total_cost") is not None else ""}
            </p>
            """, unsafe_allow_html=True)

            if not res["found"]:
                st.error("No route found — destination is unreachable with the current obstacles.")
