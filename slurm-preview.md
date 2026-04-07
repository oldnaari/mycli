I want you to create a tool `slurm-monitor` for monitoring slurm trainings:

|    PRIORITY   |                                  |
| ------------- | INFORMATION ON THE SELECTED NODE |
| LIST OF NODES |                                  |
|--------------------------------------------------|
| NVIDIA-SMI command
|--------------------------------------------------|
| STATUS BAR                                       |




the List of nodes should contain two columns: node name, node usage


# PRIORITY

Display the priority jobs 
contains two columns
1) partition [N-{num nodes} G-{num gpus}]
2) username

the columns should be aligned with LIST OF NODES

THIS part is non-interactable

Use single line text "PRIORITY" to indicate the table boundaries
There are now vertical horiznotal dividers in table
the cilumns are splitted only visually
the contents of the table should be aligned to righr

# LIST OF NODES

## NODE USAGE

the node usage should appear as follows:
it should be N_GPUS_IN_NODE symbols of blocks coloured as follows:

* RED if the GPU is drained
* YELLOW if the GPU is used it GPU UTILIZATION is more then 5%
* GREEN if he GPU is used but has low GPU utilization
* BLUE if the GPU is free
* WHITE if the nodes are used by the current user

## NODE NAME

display the node name in the following color:

* WHITE: if the node is used by the user
* BLUE: if it is free
* GREEN: if it is mixed, or it has a GREEN gpu
* RED: the node is drained
* YELLOW: if all the GPU are used

when displayed the nodes in the list should be dsiplayed in the said order

## INTERACTION

the user can use up/down arrows or jk to move the selection up/down the list
non-selected lines should have no background
the selected line should have the darkest black from the theme as background colour

one can use 
u / esc - to toggle/untaggle filtration and search by username
n / esc - to toggle/untaggle filtration and search by node name

The column should look minimalistic:
* No header for the table columns
* No divider for the table, the columns should be apparent from the tabulation/alignment only

USe a single line "LIST OF NODES" to indicate the start of the table
the contents of the tableshould be aligned to right


the lines for both priority and list of nodes should both use bgcolor bright_black, to separate from the table contents
# INFORMATION ON THE SELECTED NODE:

For each GPU on node:
display the username of user that uses the said GPU.
Have a percentage bar that displays how much vram is used 
Use the same colour for the GPU as in the node name: 
Use bold for username of the user that executes the command.

# NVIDIA_SMI COMMAND

for the selected node provide a one line command that will run and display the nvidia-smi for the said node
The purpose of this line is to create a copy-paste line, that user can use in terminal to get more detailed info if needed

# STATUS BAR

By default should show the list of command one can use to interact with the application
when starting a search the status bar becomes an input field for search

the status bar should use bright_black for background, and blue for foreground


