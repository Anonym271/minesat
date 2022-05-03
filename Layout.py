from copy import deepcopy
from enum import Enum
from typing import NamedTuple
from pathlib import Path
import yaml
from BooleanParser import BooleanParser, BooleanFormula, SyntaxTree, TreeType



ELEMENTS_DIR = Path('./elements')
GATE_CACHE = {}


Point = NamedTuple('Point', [('x', int), ('y', int)])


class ElementSizes:
    Wire = 3
    PhaseShifter1 = 7
    PhaseShifter2 = 11
    Not1 = 5
    Not3 = 9 # Triple NOT gate (leaves phase unaffected)
    BenderSEDeltaY = 5
    BenderWNDeltaY = 2
    VarpartControlElements = 12
    VerticalConnectionWidth = 7



class Direction(Enum):
    NORTH = 'n'
    SOUTH = 's'
    EAST = 'e'
    WEST = 'w'

MIRROR_DIR = {
    'n': 's',
    's': 'n',
    'e': 'w',
    'w': 'e',
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}

def mirror_dir(dir, axis):
    if axis == 'x':
        if dir == 'e' or dir == 'w' or dir == Direction.EAST or dir == Direction.WEST:
            return dir
    elif axis == 'y':
        if dir == 'n' or dir == 's' or dir == Direction.NORTH or dir == Direction.SOUTH:
            return dir
    return MIRROR_DIR[dir]



class Port:
    def __init__(self, x: int, y: int, dir: Direction | str):
        self.direction = dir if isinstance(dir, Direction) else Direction(dir)
        self.x = x
        self.y = y
    
    @property
    def position(self):
        return self.x, self.y
    
    def __repr__(self):
        return f'Port(x={self.x}, y={self.y}, dir={self.direction})'



class Element:
    def __init__(self, name: str, type: str, width=0, height=0, 
            inputs: list[Port] = None, outputs: list[Port] = None, rows: list[str] = None):
        self.name = name
        self.width = width
        self.height = height
        self.inputs = inputs if inputs != None else []
        self.outputs = outputs if outputs != None else []
        self.rows = rows if rows != None else []
        self.type = type
        self.x = 0
        self.y = 0

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            raw = yaml.safe_load(f)
        name = raw['name']
        width = raw['width']
        height = raw['height']
        inputs = [Port(**indata) for indata in raw['inputs']]
        outputs = [Port(**outdata) for outdata in raw['outputs']]
        rows = raw['rows']
        type = raw['type']
        if len(rows) != height:
            raise ValueError('The row count is not the same as the defined height')
        for row in rows:
            if len(row) != width:
                raise ValueError('The row count is not the same as the defined height')
        return Element(name, type, width, height, inputs, outputs, rows)
    
    def input_coords_rel(self, xoffs=0, yoffs=0) -> list[Point]:
        return [Point(xoffs + self.x + inp.x, yoffs + self.y + inp.y) for inp in self.inputs]
    def output_coords_rel(self, xoffs=0, yoffs=0) -> list[Point]:
        return [Point(xoffs + self.x + out.x, yoffs + self.y + out.y) for out in self.outputs]
    
    @property 
    def is_unary_gate(self):
        return self.type == 'gate' and len(self.inputs) == 1
    @property
    def is_binary_gate(self):
        return self.type == 'gate' and len(self.inputs) == 2
    
    @property
    def bottom(self):
        return self.y + self.height
    @property
    def right(self):
        return self.x + self.width

    def mirror_x(self):
        rows = self.rows.copy()
        rows.reverse()
        inp = [Port(i.x, self.height - i.y, mirror_dir(i.direction, 'x')) for i in self.inputs]
        outp = [Port(o.x, self.height - o.y, mirror_dir(o.direction, 'x')) for o in self.outputs]
        return Element(self.name, self.type, self.width, self.height,
            inp, outp, rows)

    def mirror_y(self):
        rows = [row[::-1] for row in self.rows]
        inp = [Port(self.width - i.x, i.y, mirror_dir(i.direction, 'y')) for i in self.inputs]
        outp = [Port(self.width - o.x, o.y, mirror_dir(o.direction, 'y')) for o in self.outputs]
        return Element(self.name, self.type, self.width, self.height,
            inp, outp, rows)




def get_element(tree_or_name: SyntaxTree | str = None):
    if isinstance(tree_or_name, str):
        name = tree_or_name
    else:
        tree = tree_or_name
        if tree.type == TreeType.OPERATOR:
            name = tree.value

        elif tree.type == TreeType.VARIABLE:
            # Variable is a pseudo element with 0 width
            return Element(tree.value, 'var', 0, 3, [Port(0, 1, Direction.EAST)], [Port(0, 1, Direction.EAST)], ["","",""])
        else:
            # TODO
            raise NotImplementedError()
    res = GATE_CACHE.get(name)
    if res == None:
        res = Element.load(ELEMENTS_DIR / (name + '.yaml'))
        GATE_CACHE[name] = res
    return deepcopy(res)


def get_phaseshifter(number, dir):
    if number == 1:
        numstr = ''
    elif number == 2:
        numstr = '2'
    else:
        raise ValueError(f'Invalid PS number: {number}')
    if dir == 'ns':
        return get_element(f'ps{numstr}_ns')
    elif dir == 'sn':
        return get_element(f'ps{numstr}_ns').mirror_x()
    if dir == 'we':
        return get_element(f'ps{numstr}')
    elif dir == 'ew':
        return get_element(f'ps{numstr}').mirror_y()



class Cell:
    def __init__(self, i, j, element=None, tree=None, parent_coords=None, child_coords=None):
        self.element = element
        self.tree = tree
        self.i = i
        self.j = j
        self.parent_coords = parent_coords
        self.child_coords = [] if child_coords == None else child_coords
        self.ps_north = 0
        self.ps_east = 0 # PhaseShifters east
        self.ps_south = 0
        self.ps_west = 0
        self.var_flags = []
    
    def set_tree(self, tree: SyntaxTree):
        self.tree = tree
        self.element = get_element(tree)

    @property
    def width(self):
        if self.element == None:
            return 0
        return self.element.width
    
    @property
    def width_ps(self):
        w = self.width
        ps1c = self.ps_east % 2
        ps2c = self.ps_east // 2
        return w + ps1c*ElementSizes.PhaseShifter1 + ps2c * ElementSizes.PhaseShifter2

    @property
    def has_element(self):
        return self.element != None

    @property
    def total_top(self):
        """Returns the y coordinate in the row, including the output bender (if present)"""
        e = self.element
        if e == None:
            return 0
        # No parent or parent in self row => no bender on output required => top is element.top
        if self.parent_coords == None or self.parent_coords[1] == self.j:
            return e.y
        # Parent is in other row => bender required => top is min(bender.top, element.top)
        if len(e.outputs) != 1:
            raise Exception("Inconsitent grid: Element has no (or more than 1) output but cell has a parent")
        bendertop = e.y + e.outputs[0].y - ElementSizes.BenderWNDeltaY
        return min(e.y, bendertop)
    
    @property
    def total_bottom(self):
        """Returns the bottom position in the row, including the input bender (if present)"""
        e = self.element
        if e == None:
            return 0
        if e.is_binary_gate:
            benderbottom = e.y + e.inputs[1].y + ElementSizes.BenderSEDeltaY
            return max(e.bottom, benderbottom)
        return e.y + e.height



class Column(list[Cell]):
    def __init__(self, index, height=0):
        list[Cell].__init__(self, [Cell(index, y) for y in range(height)])
        self.has_vertical_junction = False


class Grid:
    """A grid that is arranged in columns"""
    def __init__(self, width=0, height=0):
        self.width = width # Number of columns (not actual width in fields)
        self.height = height # Number of rows
        self.columns: list[Column] = [Column(i, height) for i in range(width)]
        self.row_heights: list[int] = [0] * height
        self.col_widths: list[int] = [0] * width


    def __getitem__(self, key: tuple[int, int]):
        x, y = key
        return self.columns[x][y]
        
    def __setitem__(self, key: tuple[int, int], value: Cell):
        x, y = key
        self.columns[x][y] = value

    def iterrow(self, j: int):
        """Returns an iterator over all cells in the row j *that contain an element*"""
        return RowElementIterator(self, j)
    
    def itercol(self, i: int):
        """Returns an iterator over all cells in the column i *that contain an element*"""
        return ColumnElementIterator(self, i)

    def __iter__(self):
        return GridIterator(self)



class GridIterator:
    def __init__(self, grid: Grid):
        self.g = grid
        self.i = -1
        self.j = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.i += 1
        if self.i >= self.g.width:
            self.i = 0
            self.j += 1
        if self.j >= self.g.height:
            raise StopIteration()
        return self.g[self.i, self.j]


class RowElementIterator:
    def __init__(self, grid: Grid, rowIndex: int):
        self.grid: Grid = grid
        self.i = -1
        self.j = rowIndex
    
    def __iter__(self):
        return self
    
    def __next__(self) -> tuple[int, Cell]:
        while True:
            self.i += 1
            if self.i >= self.grid.width:
                raise StopIteration()
            c = self.grid[self.i, self.j]
            if c != None and c.has_element:
                return self.i, c


class ColumnElementIterator:
    def __init__(self, grid: Grid, columnIndex: int):
        self.grid = grid
        self.i = columnIndex
        self.j = -1
    
    def __iter__(self):
        return self
    
    def __next__(self) -> tuple[int, Cell]:
        while True:
            self.j += 1
            if self.j >= self.grid.height:
                raise StopIteration()
            c = self.grid[self.i, self.j]
            if c != None and c.has_element:
                return self.j, c



class Field:
    def __init__(self, width: int, height: int, default: str = ' '):
        self.rows = [[default for i in range(width)] for j in range(height)]
        self.width = width
        self.height = height
    

    def draw_element(self, e: Element, x: int, y: int):
        if x + e.width > self.width or y + e.height > self.height:
            raise IndexError('Element size out of field range')
        for rowno, row in enumerate(e.rows):
            r = list(row)
            self.rows[y + rowno][x:x+len(r)] = r
    

    def draw_on_input(self, inp: Point, element: Element, output_index=0):
        out = element.outputs[output_index]
        x = inp.x - out.x
        y = inp.y - out.y
        self.draw_element(element, x, y)
        return [Point(x + p.x, y + p.y) for p in element.inputs]


    def draw_on_output(self, outp: Point, element: Element, input_index=0):
        inp = element.inputs[input_index]
        x = outp.x - inp.x
        y = outp.y - inp.y
        self.draw_element(element, x, y)       
        return [Point(x + p.x, y + p.y) for p in element.outputs]



    def draw_ps_on_output(self, output: Point, ps_count: int, dir: str):
        if ps_count > 0:
            self.draw_on_output(output, get_phaseshifter(ps_count, dir))


    def connect_wire_we(self, start: Point, end: Point):
        """Draw a cable connection between output 'start' and input 'end'"""
        d = end.x - start.x
        ps_count, ps_len = get_phaseshifter_info(d)
        self.draw_ps_on_output(start, ps_count, 'we')
        wire = get_element('wire_we')
        pos = Point(start.x + ps_len, start.y)
        while pos.x < end.x:
            pos = self.draw_on_output(pos, wire)[0]
        assert pos.x == end.x, f"Cable connection failed: x={pos.x} vs x!={end.x}"
    
    def connect_wire_ns(self, start: Point, end: Point):
        d = end.y - start.y
        ps_count, ps_len = get_phaseshifter_info(d)
        self.draw_ps_on_output(start, ps_count, 'ns')
        wire = get_element('wire_ns')
        pos = Point(start.x, start.y + ps_len)
        while pos.y < end.y:
            pos = self.draw_on_output(pos, wire)[0]
        assert pos.y == end.y, f"Cable connection failed: y={pos.y} vs y!={end.y}"

    def connect_wire_sn(self, start: Point, end: Point):
        d = start.y - end.y
        ps_count, ps_len = get_phaseshifter_info(d)
        self.draw_ps_on_output(start, ps_count, 'sn')
        wire = get_element('wire_sn')
        pos = Point(start.x, start.y - ps_len)
        while pos.y > end.y:
            pos = self.draw_on_output(pos, wire)[0]
        assert pos.y == end.y, f"Cable connection failed: y={pos.y} vs y!={end.y}"



    def __repr__(self):
        nl = '\n'
        return f"Field({self.width}x{self.height}):{nl}{nl.join(f'{i:03}: ' + ''.join(row) for i, row in enumerate(self.rows))}"
    
    def __getitem__(self, pos: tuple[int, int]):
        return self.rows[pos[1]][pos[0]]



class FieldBuilder:
    def __init__(self, tree: SyntaxTree):
        self.tree = tree
        self.field: Field = None

    def build(self):
        self.create_grid()
        self.arrange_tree_part()
        self.arrange_variable_part()
        self.create_correction_column()
        self.align_inout_y()
        self.adjust_col_width_treepart()
        self.adjust_col_width_correction()
        self.adjust_col_width_varpart()
        self.define_row_heights()
        self.init_field()
        self.connect_tree_part()
        self.connect_var_part()
        self.connect_correction_col()
        return self.field

    
    # 1. Grid erstellen
    def create_grid(self):
        tp_width, tp_height, self.variables = get_dimensions(self.tree)
        self.variable_list = sorted(self.variables)
        self.nvars = len(self.variable_list)
        self.varpart_start = 0
        self.correction_column_index = self.nvars
        self.treepart_start = self.nvars + 1
        self.grid = Grid(self.nvars + 1 + tp_width, tp_height)
    

    # 2. Syntaxbaum im Grid anordnen
    def arrange_tree_part(self):
        self.__arrange_tree_part_impl(self.tree)
        
    def __arrange_tree_part_impl(self, tree: SyntaxTree, parent_coords: tuple[int, int] | None = None, depth=1, leaf=0):
        g = self.grid
        x = g.width - depth
        y = leaf
        if len(tree.children) == 0:
            cell = Cell(x, y, get_element(tree), tree, parent_coords)
            cell.children_coords = []
            g[x, y] = cell
            return (x, y), leaf + 1
        children_coords = []
        for child in tree.children:
            childpos, leaf = self.__arrange_tree_part_impl(child, (x, y), depth + 1, leaf)
            children_coords.append(childpos)
        cell = Cell(x, y, get_element(tree), tree, parent_coords, children_coords)
        g[x, y] = cell
        return (x, y), leaf


    # 3. Variablenteil im Grid anorgnen
    def arrange_variable_part(self):
        g = self.grid
        self.__find_var_inputs()
        for i, var in enumerate(self.variable_list):
            splitters = []
            for input_i, j in self.var_input_dict[var]:
                for current_i in range(i+1, self.correction_column_index):
                    g[current_i, j].var_flags.append(var)
                splitters.append((i, j))
            split_it = reversed(splitters)
            last_pos = next(split_it)
            g[last_pos].element = get_element('bend_ne') # Last one is a bender
            for pos in split_it:
                g[pos].element = get_element('split_nes') # The rest are splitters
            for j in range(last_pos[1]):
                g[i, j].var_flags.append(var)
        for i in range(self.correction_column_index):
            for j in range(g.height):
                if len(g[i, j].var_flags) > 1:
                    g[i, j].element = get_element('cross')

    def __find_var_inputs(self):
        self.var_inputs = []
        self.var_input_dict = {}
        for var in self.variables:
            self.var_input_dict[var] = []
        for j in range(self.grid.height):
            for i in range(self.treepart_start, self.grid.width):
                cell = self.grid[i, j]
                if cell.has_element and cell.element.type == 'var':
                    var = cell.element.name
                    self.var_inputs.append(cell)
                    self.var_input_dict[var].append((i, j))
                    break # Next row


    # 4. Korrekturspalte
    def create_correction_column(self):
        neg_col = self.grid.columns[self.correction_column_index]
        for i, v in enumerate(self.variable_list):
            neg = False
            for j, cell in self.grid.itercol(i):                    
                if neg:
                    neg_col[j].element = get_element('not')
                if cell.element.name == 'split_nes':
                    neg = not neg
                else: # Bender reached
                    break # Variable column done, next one
    

    # 5. Inputs und Outputs y-alignen
    def align_inout_y(self):
        g = self.grid
        for j in range(g.height):
            y_min = 0
            last_out_y = 0
            for i, cell in g.iterrow(j):
                e = cell.element
                in_w = get_west_in(e)
                out_e = get_east_out(e)
                if in_w != None:
                    diff = in_w.y - last_out_y
                    e.y -= diff
                last_out_y = e.y + out_e.y
                y_min = min(y_min, e.y)
            for i, cell in g.iterrow(j):
                cell.element.y -= y_min
    

    # 6. Spaltenbreite Baumteil
    def adjust_col_width_treepart(self):
        g = self.grid
        for i in range(self.treepart_start, g.width):
            max_width = max(cell.element.width for j, cell in g.itercol(i))
            has_binary = False
            for j, cell in g.itercol(i):
                if cell.element.is_binary_gate:
                    has_binary = True
                w = cell.element.width
                diff = max_width - w
                ps_count, ps_len = get_phaseshifter_info(diff)
                if ps_count > 0:
                    cell.ps_east = ps_count
                    max_width = max(max_width, ps_len)
            if has_binary:
                g.col_widths[i-1] += ElementSizes.VerticalConnectionWidth
            g.col_widths[i] = max_width
    

    # 7. Spaltenbreite Korrekturspalte
    def adjust_col_width_correction(self):
        g = self.grid
        i = self.correction_column_index
        corr = g.columns[i]
        max_width = max((cell.element.width for j, cell in g.itercol(i)), default=0)
        for j in range(g.height):
            corr_cell = corr[j]
            required = corr_cell.width
            d = max_width - corr_cell.width # Rest of the column
            in_i = self.var_inputs[j].i
            for n in range(i+1, in_i):
                d += g.col_widths[n]
            ps_count, ps_len = get_phaseshifter_info(d)
            corr_cell.ps_east = ps_count
            required += ps_len
            max_width = max(max_width, required)
        g.col_widths[i] = max_width
    

    # 8. Spaltenbreite Variablenteil
    def adjust_col_width_varpart(self):
        for i in range(self.nvars):
            self.grid.col_widths[i] = ElementSizes.VarpartControlElements
    

    # 9. Zeilenhöhe definieren
    def define_row_heights(self):
        g = self.grid
        for j in range(g.height):
            g.row_heights[j] = round_up_to(max(cell.total_bottom for i, cell in g.iterrow(j)), 3)
        # Variable part
        for i in range(self.correction_column_index):
            last_out_y = self.get_south_out_yabs(i, 0)
            last_row = 0
            current_y = g.row_heights[0]
            for j in range(1, g.height):
                if g[i, j].has_element:
                    in_y = self.get_north_in_yabs(i, j)
                    d = current_y + in_y - last_out_y
                    ps_count, ps_height = get_phaseshifter_info(d)
                    if ps_count > 0:
                        g[i, last_row].ps_south = ps_count
                        diff = ps_height - d
                        if diff > 0: # Not enough space for PS
                            diff = round_up_to(diff, 3)
                            g.row_heights[last_row] += diff
                            current_y += diff
                    if g[i, j].element.name == 'bend_ne':
                        break # Nothing to do in this column after the bender
                    last_out_y = current_y + self.get_south_out_yabs(i, j)
                    last_row = j
                current_y += g.row_heights[j]
        # Tree part (first column does not have children)
        for i in range(self.treepart_start+1, self.grid.width):
            current_y = 0
            for j in range(g.height):
                cell = g[i, j]
                gate = cell.element
                if gate != None and gate.is_binary_gate:
                    bottom = cell.element.input_coords_rel()[1][1] + ElementSizes.BenderSEDeltaY
                    child = g[cell.child_coords[1]]
                    d = g.row_heights[j] - bottom
                    d += sum(g.row_heights[j+1:child.j])
                    d += child.element.output_coords_rel()[0][1] - ElementSizes.BenderWNDeltaY
                    ps_count, ps_height = get_phaseshifter_info(d)
                    if ps_count > 0:
                        cell.ps_south = ps_count
                        diff = ps_height - d
                        if diff > 0:
                            g.row_heights[j] += round_up_to(diff, 3)
            current_y += g.row_heights[j]
        # Calculate absolute position of each row / column
        self.row_y = accumulate_0(self.grid.row_heights)
        self.col_x = accumulate_0(self.grid.col_widths)


    # 10. Spielfeld initialisieren & Elemente eintragen
    def init_field(self):
        w = sum(self.grid.col_widths)
        h = sum(self.grid.row_heights)
        f = Field(w, h, '#') # TODO: "#" -> " "
        for cell in self.grid:
            if cell.has_element:
                x = self.col_x[cell.i] + cell.element.x
                y = self.row_y[cell.j] + cell.element.y
                f.draw_element(cell.element, x, y)
        self.field = f
    
    
    # 11. Verbindungen im Baumteil
    def connect_tree_part(self):
        g = self.grid
        f = self.field
        ps1 = get_element('ps')
        bender_se = get_element('bend_se')
        bender_wn = get_element('bend_wn')
        stack: list[Cell] = [g[-1, 0]] # Init with cell on the top right = root of the tree
        while len(stack) > 0:
            current = stack.pop()
            e = current.element
            if e.type == 'var':
                continue
            i, j = current.i, current.j
            if len(current.child_coords) > 0:
                current_ins = e.input_coords_rel(self.col_x[i], self.row_y[j])
                child_i, child_j = current.child_coords[0]
                child = g[child_i, child_j]
                child_out = child.element.output_coords_rel(
                    self.col_x[child_i], self.row_y[child_j])[0]
                f.connect_wire_we(child_out, current_ins[0]) # Does the PS stuff automatically
                stack.append(child)
            if len(current.child_coords) > 1:
                # Render BenderSE at current input
                bender_se_in = f.draw_on_input(current_ins[1], bender_se)[0]
                # Fill up child cell with wires
                child_i, child_j = current.child_coords[1]
                child = g[child_i, child_j]
                child_out = child.element.output_coords_rel(
                    self.col_x[child_i], self.row_y[child_j])[0]
                cell_end = Point(self.col_x[current.i] - ElementSizes.VerticalConnectionWidth, child_out.y)
                f.connect_wire_we(child_out, cell_end)
                # Render BenderWN at cell end
                bender_wn_out = f.draw_on_output(cell_end, bender_wn)[0]
                # Connect benders with wires
                f.connect_wire_sn(bender_wn_out, bender_se_in)
                stack.append(child)


    # 12. Verbindungen im Variablenteil
    def connect_var_part(self):
        f = self.field
        # Vertical wires
        for i in range(self.correction_column_index):
            out = None
            for j, cell in self.grid.itercol(i):
                e = cell.element
                ex = self.col_x[i] + e.x
                ey = self.row_y[j] + e.y
                if out != None:
                    inp = get_north_in(e)
                    inp = Point(ex + inp.x, ey + inp.y)
                    f.connect_wire_ns(out, inp)
                if e.name == 'bend_ne':
                    break # Next column
                out = get_south_out(e)
                out = Point(ex + out.x, ey + out.y)
        # Horizontal wires
        for j in range(self.grid.height):
            out = None
            for i in range(self.correction_column_index):
                cell = self.grid[i, j]
                if cell.has_element:
                    e = cell.element
                    ex = self.col_x[i] + e.x
                    ey = self.row_y[j] + e.y
                    if out != None:
                        # e can only be a bender if out == None, so no need to check here
                        inp = get_west_in(e)
                        inp = Point(ex + inp.x, ey + inp.y)
                        f.connect_wire_we(out, inp)
                    out = get_east_out(e)
                    out = Point(ex + out.x, ey + out.y)
            if out != None:
                inp = Point(self.col_x[self.correction_column_index], out.y)
                f.connect_wire_we(out, inp)
    
    # 13. Verbindungen der Korrekturspalte
    def connect_correction_col(self):
        k = self.correction_column_index
        kx = self.col_x[k]
        for j in range(self.grid.height):
            cell = self.grid[k, j]
            inp_cell: Cell = self.var_inputs[j]
            inp_ex = self.col_x[inp_cell.i] + inp_cell.element.x
            inp_ey = self.row_y[inp_cell.j] + inp_cell.element.y
            inp = inp_cell.element.inputs[0]
            inp = Point(inp_ex + inp.x, inp_ey + inp.y)
            if cell.has_element:
                out_ex = kx + cell.element.x
                out_ey = self.row_y[j] + cell.element.y
                out = cell.element.outputs[0]
                out = Point(out_ex + out.x, out_ey + out.y)
            else:
                out = Point(kx, inp.y)
            self.field.connect_wire_we(out, inp)
            




    def get_south_out_yabs(self, i, j):
        e = self.grid[i, j].element
        if e == None or e.name == 'bend_ne':
            return 0
        return e.y + get_south_out(e).y
    
    def get_north_in_yabs(self, i, j):
        e = self.grid[i, j].element
        return e.y + get_north_in(e).y


def get_dimensions(tree: SyntaxTree, depth=1, vars=None):
    if vars == None:
        vars = set()
    if tree.type == TreeType.OPERATOR:
        leaves = 0
        maxDepth = depth
        for c in tree.children:
            d, n, vars = get_dimensions(c, depth + 1, vars)
            if d > maxDepth:
                maxDepth = d
            leaves += n
        return maxDepth, leaves, vars
    elif tree.type == TreeType.VARIABLE:
        vars.add(tree.value)
    return depth, 1, vars

def round_up_to(x: int, target: int):
    """Rounds x up to the next n*target"""
    rest = x % target
    if rest != 0:
        x += target - rest
    return x

def get_south_out(e: Element):
    if e.name == 'cross' or e.name == 'split_nes':
        return e.outputs[0]
    raise ValueError('Illegal element for get_south_out: ' + e.name)

def get_north_in(e: Element):
    if e.name == 'cross' or e.name == 'split_nes' or e.name == 'bend_ne':
        return e.inputs[0]
    raise ValueError('Illegal element for get_north_in: ' + e.name)

def get_west_in(e: Element):
    if e.name == 'split_nes' or e.name == 'bend_ne':
        return None
    if e.name == 'cross':
        return e.inputs[1]
    return e.inputs[0]

def get_east_out(e: Element):
    if e.name == 'cross' or e.name == 'split_nes':
        return e.outputs[1]
    return e.outputs[0]

def get_phaseshifter_info(d: int) -> tuple[int, int]:
    """Returns information (PS count, required space) about the 
    required phaseshifters for a distance d"""
    rest = d % 3
    if rest == 0:
        return 0, 0
    elif rest == 1:
        return 1, ElementSizes.PhaseShifter1
    else: # rest == 2
        return 2, ElementSizes.PhaseShifter2

def accumulate_0(items):
    res = [0] * len(items)
    for i in range(len(res)-1):
        res[i+1] = res[i] + items[i]
    return res


if __name__ == "__main__":
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument("formula")
    args = args.parse_args()

    parser = BooleanParser(args.formula)
    tree = parser.parse()

    builder = FieldBuilder(tree)
    f = builder.build()
    
    print(f)
    pass