from nbformat import read, write

notebook_path = "data_processing.ipynb"  # change this to your actual filename

with open(notebook_path) as f:
    nb = read(f, as_version=4)

for i, cell in enumerate(nb.cells):
    if cell.cell_type == "markdown":
        if i == 0:
            cell.metadata['slideshow'] = {'slide_type': 'slide'}
        else:
            cell.metadata['slideshow'] = {'slide_type': 'subslide'}
    elif cell.cell_type == "code":
        cell.metadata['slideshow'] = {'slide_type': 'subslide'}  # or '-' if you want to hide

with open(notebook_path, "w") as f:
    write(nb, f)
