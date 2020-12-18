from matplotlib import pyplot as plt
from IPython.display import display, Markdown

def show_batch_of_data(x, y):
    display(Markdown(f"**targets:** {y}"))
    plt.imshow(x.transpose(1,0).reshape(-1,x.shape[0]*28)); plt.show()

def show_first_n_batches(dl, n):
    for i, (x, y) in enumerate(dl):
        if i >= n: break
        show_batch_of_data(x, y)

