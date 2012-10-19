import matplotlib.pyplot as plt

def plot_color_histogram(hist):
    fig, axarr = plt.subplots(3, sharex=True, sharey=True)
    axarr[0].fill_between(hist['red'][0], 0, hist['red'][1],
                     facecolor='red')
    axarr[1].fill_between(hist['green'][0], 0, hist['green'][1],
                     facecolor='green')
    axarr[2].fill_between(hist['blue'][0], 0, hist['blue'][1],
                     facecolor='blue')
    axarr[0].set_xlim(0,256)
    axarr[1].set_ylim(ymin=0)
    fig.show()
