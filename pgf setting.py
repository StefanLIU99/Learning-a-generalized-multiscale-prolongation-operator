import matplotlib.pyplot as plt
import matplotlib as mpl


plt.style.use("seaborn-v0_8-paper")

LARGE_FONT_SIZE = 9
DEFAULT_FONT_SIZE = 9
SMALL_FONT_SIZE = 9

mpl.rcParams["font.size"] = DEFAULT_FONT_SIZE
mpl.rcParams["axes.titlesize"] = DEFAULT_FONT_SIZE
mpl.rcParams["axes.labelsize"] = DEFAULT_FONT_SIZE
mpl.rcParams["xtick.labelsize"] = SMALL_FONT_SIZE
mpl.rcParams["ytick.labelsize"] = SMALL_FONT_SIZE
mpl.rcParams["legend.fontsize"] = SMALL_FONT_SIZE
mpl.rcParams["figure.titlesize"] = LARGE_FONT_SIZE
mpl.rcParams["pgf.preamble"] = "\n".join(
    [
        r"\usepackage{bm}%",
        r"\usepackage{mismath}",
        r"\newcommand{\SSSText}[1]{{\scriptscriptstyle \mathup{#1}}}%",
        r"\renewcommand{\mathdefault}[1][]{}%",
    ]
)


# mpl.use('pgf')
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex", 
    "text.usetex": True,          
    "font.family": "serif",       
    "font.serif": [],             
    "pgf.rcfonts": False,         
})
text_width = 5.3
fig_width = text_width 
fig_height = text_width / 2.6
dpi = 1000

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times'
mpl.rcParams['text.usetex'] = True 

plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
