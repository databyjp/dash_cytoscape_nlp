# ========== (c) JP Hwang 2020-04-02  ==========

import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import pandas as pd
import numpy as np
import plotly.express as px
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output
from flask_caching import Cache
from sklearn.manifold import TSNE
import umap
from copy import deepcopy
import os
import json

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
#
# cache_dir = "./cache"
# if not os.path.exists(cache_dir):
#     os.mkdir(cache_dir)
#
# cache = Cache(
#     app.server,
#     config={
#         # try 'filesystem' if you don't want to setup redis
#         "CACHE_TYPE": "redis",
#         "CACHE_REDIS_URL": os.environ.get("REDIS_URL", ""),
#     },
# )

# network_df = pd.read_csv('outputs/network_df.csv', index_col=0)  # ~8300 nodes
network_df = pd.read_csv("outputs/network_df_sm.csv", index_col=0)  # ~4700 nodes

# Prep data / fill NAs
network_df["citations"] = network_df["citations"].fillna("")
network_df["cited_by"] = network_df["cited_by"].fillna("")
network_df["topic_id"] = network_df["topic_id"].astype(str)
topic_ids = [str(i) for i in range(len(network_df["topic_id"].unique()))]
# lda_val_arr = network_df[topic_ids].values

with open("outputs/lda_topics.json", "r") as f:
    lda_topics = json.load(f)
topics_txt = [lda_topics[str(i)] for i in range(len(lda_topics))]
topics_txt = [[j.split("*")[1].replace('"', "") for j in i] for i in topics_txt]
topics_txt = ["; ".join(i) for i in topics_txt]

journal_ser = network_df.groupby("journal")["0"].count().sort_values(ascending=False)
top_journals = list(journal_ser.index[:3])


def tsne_to_cyto(tsne_val, scale_factor=40):
    return int(scale_factor * (float(tsne_val)))


network_df = network_df.assign(highlight=1)


def get_node_list(in_df=network_df):  # Convert DF data to node list for cytoscape
    return [
        {
            "data": {
                "id": str(i),
                "label": str(i),
                "title": row["title"],
                "journal": row["journal"],
                "pub_date": row["pub_date"],
                "authors": row["authors"],
                "cited_by": row["cited_by"],
                "n_cites": row["n_cites"],
                "node_size": int(np.sqrt(1 + row["n_cites"]) * 10),
                "highlight": row["highlight"],
            },
            "position": {"x": tsne_to_cyto(row["x"]), "y": tsne_to_cyto(row["y"])},
            "classes": row["topic_id"],
            "selectable": True,
            "grabbable": False,
        }
        for i, row in in_df.iterrows()
    ]


node_list = get_node_list()


# @cache.memoize()  # Caching node location results where they remain identical, as they are time consuming to calculate
def get_node_locs(in_df, dim_red_algo="tsne", tsne_perp=40):
    logger.info(f"Starting dimensionality reduction on {len(in_df)} nodes, with {dim_red_algo}")

    if dim_red_algo == "tsne":
        node_locs = TSNE(
            n_components=2,
            perplexity=tsne_perp,
            n_iter=300,
            n_iter_without_progress=100,
            learning_rate=150,
            random_state=23,
        ).fit_transform(in_df[topic_ids].values)
    elif dim_red_algo == "umap":
        reducer = umap.UMAP(n_components=2)
        node_locs = reducer.fit_transform(in_df[topic_ids].values)
    else:
        logger.error(
            f"Dimensionality reduction algorithm {dim_red_algo} is not a valid choice! Something went wrong"
        )
        node_locs = np.zeros([len(in_df), 2])

    logger.info("Finished dimensionality reduction")

    x_list = node_locs[:, 0]
    y_list = node_locs[:, 1]

    return x_list, y_list


default_tsne = 40


# (x_list, y_list) = get_node_locs(network_df, tsne_perp=default_tsne)


def update_node_data(
        node_bools, dim_red_algo, tsne_perp, node_list_in=node_list, in_df=network_df
):
    # node_list_in = deepcopy(node_list)
    (x_list, y_list) = get_node_locs(in_df, dim_red_algo, tsne_perp=tsne_perp)

    x_range = max(x_list) - min(x_list)
    y_range = max(y_list) - min(y_list)
    # print("Ranges: ", x_range, y_range)

    scale_factor = int(4000 / (x_range + y_range))

    for i in range(len(in_df)):
        tempbool = node_bools[i]
        node_list_in[i]["data"]["highlight"] = tempbool
        node_list_in[i]["selectable"] = False if tempbool == 0 else True

        node_list_in[i]["position"]["x"] = tsne_to_cyto(x_list[i], scale_factor)
        node_list_in[i]["position"]["y"] = tsne_to_cyto(y_list[i], scale_factor)

    return node_list_in


def draw_edges(in_df=network_df):
    conn_list_out = list()

    for i, row in in_df.iterrows():
        citations = row["cited_by"]

        if len(citations) == 0:
            citations_list = []
        else:
            citations_list = citations.split(",")

        for cit in citations_list:
            if int(cit) in in_df.index:
                tgt_topic = row["topic_id"]
                temp_dict = {
                    "data": {"source": cit, "target": str(i)},
                    "classes": tgt_topic,
                    "tgt_topic": tgt_topic,
                    "src_topic": in_df.loc[int(cit), "topic_id"],
                    "locked": True,
                }
                conn_list_out.append(temp_dict)

    return conn_list_out


elm_list = node_list

col_swatch = px.colors.qualitative.Dark24
def_stylesheet = [
    {
        "selector": "." + str(i),
        "style": {"background-color": col_swatch[i], "line-color": col_swatch[i]},
    }
    for i in range(len(network_df["topic_id"].unique()))
]
def_stylesheet += [
    {
        "selector": "node",
        "style": {"width": "data(node_size)", "height": "data(node_size)"},
    },
    {"selector": "node[highlight = 0]", "style": {"background-opacity": 0.15}},
    {"selector": "edge", "style": {"width": 1, "curve-style": "bezier"}},
]

navbar = dbc.NavbarSimple(
    brand="Plotly dash-cytoscape demo - CORD-19 LDA analysis output",
    brand_href="#",
    color="dark",
    dark=True,
)

topics_html = list()
for topic_html in [
    html.Span([str(i) + ": " + topics_txt[i]], style={"color": col_swatch[i]})
    for i in range(len(topics_txt))
]:
    topics_html.append(topic_html)
    topics_html.append(html.Br())

body_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Markdown(
                            f"""
                -----
                ##### Data:
                -----
                For this demonstration, {len(network_df)} papers from the CORD-19 dataset* were categorised into
                {len(network_df.topic_id.unique())} topics using
                [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) analysis.

                Each topic is shown in different color on the citation map, as shown on the right.
                """
                        )
                    ],
                    sm=12,
                    md=4,
                ),
                dbc.Col(
                    [
                        dcc.Markdown(
                            """
                -----
                ##### Topics:
                -----
                """
                        ),
                        html.Div(
                            topics_html,
                            style={
                                "fontSize": 11,
                                "height": "100px",
                                "overflow": "auto",
                            },
                        ),
                    ],
                    sm=12,
                    md=8,
                ),
            ]
        ),
        dbc.Row(
            [
                dcc.Markdown(
                    """
            -----
            ##### Filter / Explore node data
            Node size indicates number of citations from this collection, and color indicates its
            main topic group.

            Use these filters to highlight papers with:
            * certain numbers of citations from this collection, and
            * by journal title

            Try showing or hiding citation connections with the toggle button, and explore different visualisation options.

            -----
            """
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                cyto.Cytoscape(
                                    id="core_19_cytoscape",
                                    layout={"name": "preset"},
                                    style={"width": "100%", "height": "400px"},
                                    elements=elm_list,
                                    stylesheet=def_stylesheet,
                                    minZoom=0.06,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Alert(
                                    id="node-data",
                                    children="Click on a node to see its details here",
                                    color="secondary",
                                )
                            ]
                        ),
                    ],
                    sm=12,
                    md=8,
                ),
                dbc.Col(
                    [
                        dbc.Badge(
                            "Minimum citation(s):", color="info", className="mr-1"
                        ),
                        dbc.FormGroup(
                            [
                                dcc.Dropdown(
                                    id="n_cites_dropdown",
                                    options=[
                                        {"label": k, "value": k} for k in range(1, 21)
                                    ],
                                    clearable=False,
                                    value=3,
                                    style={"width": "50px"},
                                )
                            ]
                        ),
                        dbc.Badge(
                            "Journal(s) published:", color="info", className="mr-1"
                        ),
                        dbc.FormGroup(
                            [
                                dcc.Dropdown(
                                    id="journals_dropdown",
                                    options=[
                                        {
                                            "label": i
                                                     + " ("
                                                     + str(v)
                                                     + " publication(s))",
                                            "value": i,
                                        }
                                        for i, v in journal_ser.items()
                                    ],
                                    value=top_journals,
                                    multi=True,
                                    style={"width": "100%"},
                                ),
                            ]
                        ),
                        dbc.Badge("Citation network:", color="info", className="mr-1"),
                        dbc.FormGroup(
                            [
                                dbc.Container(
                                    [
                                        dbc.Checkbox(
                                            id="show_edges_radio",
                                            className="form-check-input",
                                            checked=True,
                                        ),
                                        dbc.Label(
                                            "Show citation connections",
                                            html_for="show_edges_radio",
                                            className="form-check-label",
                                            style={
                                                "color": "DarkSlateGray",
                                                "fontSize": 12,
                                            },
                                        ),
                                    ]
                                )
                            ]
                        ),
                        dbc.Badge(
                            "Dimensionality reduction algorithm",
                            color="info",
                            className="mr-1",
                        ),
                        dbc.FormGroup(
                            [
                                dcc.RadioItems(
                                    id="dim_red_algo",
                                    options=[
                                        {"label": "UMAP", "value": "umap"},
                                        {"label": "t-SNE", "value": "tsne"},
                                    ],
                                    value="tsne",
                                    labelStyle={
                                        "display": "inline-block",
                                        "color": "DarkSlateGray",
                                        "fontSize": 12,
                                        "margin-right": "10px",
                                    },
                                )
                            ]
                        ),
                        dbc.Badge(
                            "t-SNE parameters (not applicable to UMAP):",
                            color="info",
                            className="mr-1",
                        ),
                        dbc.Container(
                            "Current perplexity: 40 (min: 10, max:100)",
                            id="tsne_para",
                            style={"color": "DarkSlateGray", "fontSize": 12},
                        ),
                        dbc.FormGroup(
                            [
                                dcc.Slider(
                                    id="tsne_perp",
                                    min=10,
                                    max=100,
                                    step=1,
                                    marks={10: "10", 100: "100", },
                                    value=40,
                                ),
                                # html.Div(id='slider-output')
                            ]
                        ),
                    ],
                    sm=12,
                    md=4,
                ),
            ]
        ),
        dbc.Row(
            [
                dcc.Markdown(
                    """
            \* 'Commercial use subset' of the CORD-19 dataset from
            [Semantic Scholar](https://pages.semanticscholar.org/coronavirus-research)
            used, downloaded on 2/Apr/2020. The displayed nodes exclude papers that do not
            cite and are not cited by others in this set.

            \* Data analysis carried out for demonstration of data visualisation purposes only.
            """
                )
            ],
            style={"fontSize": 11, "color": "gray"},
        ),
    ],
    style={"marginTop": 20},
)

app.layout = html.Div([navbar, body_layout])


@app.callback(
    dash.dependencies.Output("tsne_para", "children"),
    [dash.dependencies.Input("tsne_perp", "value")],
)
def update_output(value):
    return f"Current t-SNE perplexity: {value} (min: 10, max:100)"


@app.callback(
    Output("core_19_cytoscape", "elements"),
    [
        Input("n_cites_dropdown", "value"),
        Input("journals_dropdown", "value"),
        Input("show_edges_radio", "checked"),
        Input("dim_red_algo", "value"),
        Input("tsne_perp", "value"),
    ],
)
def filter_nodes(usr_min_cites, usr_journals_list, show_edges, dim_red_algo, tsne_perp):
    # # Previous logic:
    # # Get node booleans based on filter
    # # Update node data by turning them on / off
    # node_bools = filter_node_data(min_conns=usr_min_cites, journals=usr_journals_list, date_filter=None)
    # node_list = update_node_data(node_bools, dim_red_algo, tsne_perp)

    # New logic:
    # Update base DF based on filter (all nodes are 'on')
    # Generate node list
    cur_df = network_df[(network_df.n_cites >= usr_min_cites)]
    if usr_journals_list is not None and usr_journals_list != []:
        cur_df = cur_df[(cur_df.journal.isin(usr_journals_list))]

    cur_node_list = get_node_list(cur_df)
    node_bools = np.ones(len(cur_df))
    cur_node_list = update_node_data(
        node_bools, dim_red_algo, tsne_perp, node_list_in=cur_node_list, in_df=cur_df
    )
    len(node_bools)
    conn_list = []

    if show_edges:
        conn_list = draw_edges(cur_df)

    elm_list = cur_node_list + conn_list

    return elm_list


@app.callback(
    Output("node-data", "children"), [Input("core_19_cytoscape", "selectedNodeData")]
)
def display_nodedata(datalist):
    contents = "Click on a node to see its details here"
    if datalist is not None:
        if len(datalist) > 0:
            data = datalist[-1]
            contents = []
            contents.append(html.H5("Title: " + data["title"].title()))
            contents.append(
                html.P(
                    "Journal: "
                    + data["journal"].title()
                    + ", Published: "
                    + data["pub_date"]
                )
            )
            contents.append(
                html.P(
                    "Author(s): "
                    + str(data["authors"])
                    + ", Citations: "
                    + str(data["n_cites"])
                )
            )

    return contents


if __name__ == "__main__":
    app.run_server(debug=True)
