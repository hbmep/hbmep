EXPERIMENT = "L_CIRC"

MAP = {
    '-C6LC': '-C',
    '-C6LE': '-E',
	'-C6LN': '-N',
	'-C6LNE': '-NE',
	'-C6LNW': '-NW',
    '-C6LS': '-S',
	'-C6LSE': '-SE',
	'-C6LSW': '-SW',
	'-C6LW': '-W',
    'C6LE-C6LW': 'E-W',
	'C6LNE-C6LSW': 'NE-SW',
	'C6LS-C6LN': 'S-N',
	'C6LSE-C6LNW': 'SE-NW',
    'C6LE-C6LC': 'E-C',
	'C6LN-C6LC': 'N-C',
	'C6LNE-C6LC': 'NE-C',
	'C6LNW-C6LC': 'NW-C',
    'C6LS-C6LC': 'S-C',
	'C6LSE-C6LC': 'SE-C',
	'C6LSW-C6LC': 'SW-C',
	'C6LW-C6LC': 'W-C',
}
VERTICES = [
    '-C',
    '-E', '-N', '-NE', '-NW',
    '-S', '-SE', '-SW', '-W'
]
DIAM = [
    'E-W', 'NE-SW', 'S-N', 'SE-NW'
]
RADII = [
    'E-C', 'N-C', 'NE-C', 'NW-C',
    'S-C', 'SE-C', 'SW-C', 'W-C',
]
