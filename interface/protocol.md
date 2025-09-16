# UAI (Universal Azul Interface)

UAI is based on UCI (Universal Chess Interface) and aims to fully represent the range of possible actions necessary
to play Azul. Potential GUIs or engines looking to interface will need to implement the following features: 

- [Commands](#commands)
- [Move Format](#move-format)
- [AzulFEN](/interface/azulfen.md)

## Commands

UAI supports a range of commands. It is recommended for interfacing programs to implement all commands, 
however not all are stricly necessary. Required commands are marked with an asterisk (*).  

Below is a table of commands that programs can expect to receive from the interface. 

| Command  | Description                            |
| -------- | -------                                |
| quit     | Quit the program as soon as possible.  |
| uai      | TODO                                   |

Below is a table of commands that the interface expects to receive from linked programs.

| Command  | Description                            |
| -------- | -------                                |
| TODO     | Description                            |


## Move Format

The move format is standardized as three two digit components, comprising a six digit number 
in the format `bowl tile_type row`. 
ex. 040102 would correspond to the fourth bowl, first tile type, and second row of our own board. 

Special exceptions: 
- Bowl 00 corresponds to the centre area
- Row 00 corresponds to the floor (penalty) row
