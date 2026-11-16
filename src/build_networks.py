"""Build the preliminary emergency and elective care-unit transfer networks.

Run from the repo root (after src.clean_data):   python3 -m src.build_networks

Network design (from the proposal): one directed, weighted, one-mode network per
cohort. Nodes = care units; edge A -> B = a patient moved from unit A to unit B
within an admission; weight = number of such moves; no self-loops.

Outputs:
    results/degree_by_unit.csv    in/out-degree for every unit in each network
    results/network_summary.csv   size and average degree of each network
    figures/networks_side_by_side.png
    figures/degree_comparison.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import networkx as nx
import numpy as np
import pandas as pd

from src.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR

COHORTS = ["emergency", "elective"]
COLORS = {"emergency": "#2a78d6", "elective": "#eb6834"}  # categorical slots 1-2
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# Units where admissions begin (see data/README.md). Used for the
# "without entry units" version of the summary.
ENTRY_UNITS = ["Emergency Department", "Discharge Lounge"]

# Short labels for figures
SHORT = {
    "Emergency Department": "ED",
    "Emergency Department Observation": "ED Obs",
    "Discharge Lounge": "Disch. Lounge",
    "Hematology/Oncology": "Heme/Onc",
    "Hematology/Oncology Intermediate": "Heme/Onc Int.",
    "Medicine/Cardiology": "Med/Cardiology",
    "Medicine/Cardiology Intermediate": "Med/Cardio Int.",
    "Cardiology Surgery Intermediate": "Cardiac Surg Int.",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro SICU",
}

# Order around the circle: entry points, ICUs, step-down, wards, other
GROUP_ORDER = ["entry", "icu", "intermediate", "ward", "other"]


def short(unit):
    if unit in SHORT:
        return SHORT[unit]
    if "(" in unit:  # e.g. "Medical Intensive Care Unit (MICU)" -> "MICU"
        return unit[unit.index("(") + 1 : unit.index(")")]
    return unit


def unit_group(unit):
    if unit in ENTRY_UNITS or unit in ("Emergency Department Observation", "PACU"):
        return "entry"
    if "Intensive Care" in unit or "ICU" in unit or "Coronary Care" in unit:
        return "icu"
    if "Intermediate" in unit or "Stepdown" in unit:
        return "intermediate"
    if unit in ("Observation", "Unknown", "Psychiatry"):
        return "other"
    return "ward"


def build_network(edges):
    """Directed weighted graph from a transitions table."""
    w = edges.groupby(["from_unit", "to_unit"]).size().reset_index(name="weight")
    G = nx.DiGraph()
    for r in w.itertuples(index=False):
        G.add_edge(r.from_unit, r.to_unit, weight=r.weight)
    return G


def degree_table(G, cohort):
    return pd.DataFrame(
        {
            "cohort": cohort,
            "careunit": list(G.nodes),
            "in_degree": [G.in_degree(n) for n in G.nodes],
            "out_degree": [G.out_degree(n) for n in G.nodes],
            "in_strength": [G.in_degree(n, weight="weight") for n in G.nodes],
            "out_strength": [G.out_degree(n, weight="weight") for n in G.nodes],
        }
    )


def summarize(G, adm, cohort, version):
    deg = degree_table(G, cohort)
    return {
        "cohort": cohort,
        "version": version,
        "admissions": adm.loc[adm.cohort == cohort, "hadm_id"].nunique(),
        "patients": adm.loc[adm.cohort == cohort, "subject_id"].nunique(),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "transfers": int(sum(d["weight"] for *_, d in G.edges(data=True))),
        "density": round(nx.density(G), 3),
        "mean_degree": round(deg.in_degree.mean(), 2),  # mean in = mean out
        "median_in_degree": deg.in_degree.median(),
        "median_out_degree": deg.out_degree.median(),
        "max_in_degree": deg.in_degree.max(),
        "max_out_degree": deg.out_degree.max(),
    }


def circle_positions(units):
    units = sorted(units, key=lambda u: (GROUP_ORDER.index(unit_group(u)), u))
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(units), endpoint=False)
    return {u: np.array([np.cos(a), np.sin(a)]) for u, a in zip(units, angles)}


def draw_network(ax, G, pos, all_units, cohort, title):
    color = COLORS[cohort]
    # Units not in this network: faint outline so the two panels line up
    absent = [u for u in all_units if u not in G]
    nx.draw_networkx_nodes(ax=ax, G=nx.Graph(), pos=pos, nodelist=absent,
                           node_size=40, node_color="white", edgecolors=GRID,
                           linewidths=1)
    weights = np.array([d["weight"] for *_, d in G.edges(data=True)])
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=0.6 + 2.4 * weights / weights.max(),
        edge_color=color, alpha=0.45, arrows=True, arrowstyle="-|>",
        arrowsize=8, connectionstyle="arc3,rad=0.12", node_size=200,
    )
    tot = np.array([G.in_degree(n) + G.out_degree(n) for n in G.nodes])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60 + 28 * tot,
                           node_color=color, edgecolors="white", linewidths=2)
    for u, (x, y) in pos.items():
        present = u in G
        if abs(x) < 0.05:  # node at the very top or bottom: label above/below
            ha, va, r = "center", ("bottom" if y > 0 else "top"), 1.09
        else:  # otherwise label to the outside, nudged up/down near the poles
            ha = "left" if x > 0 else "right"
            va = "bottom" if y > 0.9 else "top" if y < -0.9 else "center"
            r = 1.1
        ax.text(r * x, r * y, short(u), ha=ha, va=va,
                fontsize=7.5, color=TEXT if present else GRID)
    ax.set_title(title, fontsize=12, color=TEXT, loc="left", pad=12)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.22, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_degree_comparison(deg, path):
    """Dot plot: in- and out-degree of each unit, emergency vs elective."""
    units = (deg.groupby("careunit")[["in_degree", "out_degree"]].max().sum(axis=1)
             .sort_values().index.tolist())
    y = {u: i for i, u in enumerate(units)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.28 * len(units) + 1.4), sharey=True)
    for ax, col, label in zip(axes, ["in_degree", "out_degree"],
                              ["In-degree (distinct units sending patients in)",
                               "Out-degree (distinct units receiving patients)"]):
        for off, cohort in zip([0.15, -0.15], COHORTS):
            d = deg[deg.cohort == cohort]
            ax.scatter(d[col], [y[u] + off for u in d.careunit], s=34,
                       color=COLORS[cohort], edgecolors="white", linewidths=1,
                       label=cohort.capitalize(), zorder=3)
        ax.set_title(label, fontsize=10, color=TEXT, loc="left")
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlim(-0.5, deg[["in_degree", "out_degree"]].max().max() + 1)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_yticks(range(len(units)), [short(u) for u in units])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2,
               loc="upper right", bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Care-unit degree by cohort (units absent from a cohort have no dot)",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    edges = pd.read_csv(DATA_DIR / "transitions.csv")
    adm = pd.read_csv(DATA_DIR / "admissions_clean.csv")

    graphs, degs, summary = {}, [], []
    for cohort in COHORTS:
        e = edges[edges.cohort == cohort]
        G = build_network(e)
        graphs[cohort] = G
        degs.append(degree_table(G, cohort))
        summary.append(summarize(G, adm, cohort, "all units"))
        G_no_entry = build_network(
            e[~e.from_unit.isin(ENTRY_UNITS) & ~e.to_unit.isin(ENTRY_UNITS)]
        )
        summary.append(summarize(G_no_entry, adm, cohort, "without ED & Discharge Lounge"))

    deg = pd.concat(degs).sort_values(["cohort", "in_degree", "out_degree"],
                                      ascending=[True, False, False])
    summary = pd.DataFrame(summary)
    deg.to_csv(RESULTS_DIR / "degree_by_unit.csv", index=False)
    summary.to_csv(RESULTS_DIR / "network_summary.csv", index=False)

    # Network figure, shared node positions
    all_units = set(graphs["emergency"]) | set(graphs["elective"])
    pos = circle_positions(all_units)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, cohort in zip(axes, COHORTS):
        s = summary[(summary.cohort == cohort) & (summary.version == "all units")].iloc[0]
        draw_network(ax, graphs[cohort], pos, all_units, cohort,
                     f"{cohort.capitalize()} admissions  ·  {s.admissions} admissions, "
                     f"{s.nodes} units, {s.edges} edges")
    fig.text(0.01, 0.02, "Node size = in-degree + out-degree. Edge width = number of "
             "transfers. Grey labels = unit not used by this cohort. "
             "MIMIC-IV demo v2.2.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES_DIR / "networks_side_by_side.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    plot_degree_comparison(deg, FIGURES_DIR / "degree_comparison.png")

    print(summary.to_string(index=False))
    print()
    print(deg.groupby("cohort").head(6).to_string(index=False))


if __name__ == "__main__":
    main()
