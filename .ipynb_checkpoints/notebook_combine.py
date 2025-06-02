import nbformat

with open('task1_preprocessing.ipynb') as f1, open('combined_notebook.ipynb') as f2:
    nb1 = nbformat.read(f1, as_version=4)
    nb2 = nbformat.read(f2, as_version=4)

combined_nb = nbformat.v4.new_notebook()
combined_nb.cells = nb1.cells + nb2.cells

combined_nb.metadata = nb1.metadata

with open('submission.ipynb', 'w') as f:
    nbformat.write(combined_nb, f)

