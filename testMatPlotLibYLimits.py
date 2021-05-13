import matplotlib.pyplot as plt

fig, plots = plt.subplots(nrows=2, ncols=2)
top_row_plots = plots[0]

top_row_plots[0].set_xlabel('Testing')
top_row_plots[0].set_ylim((0, 2))
top_row_plots[0].plot([0, 2, 4], [.2, 1, 9])

plt.tight_layout()
plt.show()