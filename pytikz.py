# A few utility functions for printing data in the
# form of a tikz table

def addplot_table(x:list, y:list, legend=None, color="red"):
    """legend=string describing the plot"""
    assert len(x) == len(y)
    print("\\addplot [color=" + color + ", line width=1.1pt]")
    print("\ttable[row sep=crcr]{%%")
    for i in range(len(x)):
        print(str(x[i]) + " " + str(y[i]) + "\\\\")
    print("};")
    
    if legend is not None:
        print("\\addlegendentry{" + legend + "}")
    
    
def print_generic_axis(title:str="Title", xlabel:str="x", ylabel:str="y", xlog:bool=True, ylog:bool=True):
    print("\\begin{axis}[%")
    print("title=" + title + ",")
    print("xlabel={" + xlabel + "},")
    print("ylabel={" + ylabel + "},")

    comment =  """%width=1\\columnwidth,\n%height=10cm,\n%this will apply width and hight to axis, ignoring labels/title\n%scale only axis,\n%xmin=1e-3,\n%xmax=2e1,\n%ymin=1e-3,\n%ymax=2e1,"""
    print(comment)
    if xlog:
        print("xmode=log,")
    if ylog:
        print("ymode=log,")

    print("legend pos=north west,")
    print("legend style={anchor=north west, at={(0.05,0.95)},")
    print("legend cell align=left, font=\\scriptsize},")
    print("axis lines=left, grid=both, grid style={draw=black!12},\n]")
    print("\n%how the turn tables. Don't forget \\usepackage{pgfplots}\n")
    print("\\end{axis}")


## Examples ##
# print_generic_axis()
# addplot_table(a,a, legend="hi")