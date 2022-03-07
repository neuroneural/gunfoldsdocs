import graph_tools as gt
from graph_tools import draw as gtd
import numpy as np

def lg2gt(g):
    gr = gt.Graph()
    vlabel = gr.new_vertex_property("string")
    verts = {}
    edges = {}
    for v in g:
        verts[v] = gr.add_vertex()
        vlabel[verts[v]] = str(v)
    gr.vertex_properties["label"] = vlabel
    for v in g:
        for w in g[v]:
            edges[(v,w)] = gr.add_edge(verts[v], verts[w])
    return gr

def gt2g(g):
    mg = {}
    for v in g.vertices():
        if v.out_edges():
            mg[int(str(v))+1] = {int(str(x.target()))+1:1 for x in v.out_edges()}
    return mg

def plotg(g, layout='sfdp', pos=True):
    gg = lg2gt(g)
    if not pos:
        if layout=='fr':
            pos = gtd.fruchterman_reingold_layout(gg)
        else:
            pos = gtd.sfdp_layout(gg)
    else:
        pos = gg.new_vertex_property("vector<double>")
        n = gg.num_vertices()
        s = 2.0*np.pi/n
        for v in range(gg.num_vertices()):
            idx = int(gg.vertex_properties['label'][gg.vertex(v)]) - 1
            pos[gg.vertex(v)] = (n * np.cos(s * idx),
                                 n * np.sin(s * idx))

    gtd.graph_draw(gg, pos,
               vertex_text=gg.vertex_properties['label'],
               vertex_font_size=32,
               edge_pen_width=1,
               edge_marker_size=15,
               vertex_pen_width=1,
               vertex_fill_color=[0.62109375,
                                  0.875     ,
                                  0.23828125,
                                  1])
