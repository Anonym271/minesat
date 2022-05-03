from enum import Enum


operatorDict = { 
    '&': 'and',
    '*': 'and',
    '|': 'or',
    '+': 'or',
    '~': 'not',
    '-': 'not',
    '!': 'not',
    '^': 'xor'
}

inverseOperatorDict = {
    'not': '~',
    'or': 'or',
    'and': 'and',
    'xor': 'xor',
}

binaryOperatorFunctions = {
    'or': lambda x, y: x or y,
    'and': lambda x, y: x and y,
    'xor': lambda x, y: (x or y) and not (x and y),
}

operatorPrecedence = ['or', 'xor', 'and', 'not']



class TreeType(Enum):
    OPERATOR = 0
    CONSTANT = 1
    VARIABLE = 2



class BooleanLexer:
    delims = [' ', '\t', '\n', '\r']

    def __init__(self, expr):
        self.expression = expr
        self.position = 0
    
    def hasNext(self):
        while self.position < len(self.expression) and self.expression[self.position] in self.delims:
            self.position += 1
        return self.position != len(self.expression)
    
    def next(self):
        if not self.hasNext():
            return { 'type': 'EOF'}

        current = self.expression[self.position]
        self.position += 1

        # Return operators immediately
        if current == '(':
            return { 'type': '(' }
        if current == ')':
            return { 'type': ')'}

        op = operatorDict.get(current)
        if op:
            return { 'type': 'operator', 'value': op }
        
        # Check for consts
        if current.isdigit():
            numstr = current
            while self.position < len(self.expression) and self.expression[self.position].isdigit():
                numstr += self.expression[self.position]
                self.position += 1
            num = int(numstr)
            if num < 0 or num > 1:
                raise Exception(f'{num} is not a boolean value')
            return { 'type': 'const', 'value': bool(num) }

        # Get variable name
        if current.isalpha():
            varname = current
            while self.position < len(self.expression) and self.expression[self.position].isalnum():
                varname += self.expression[self.position]
                self.position += 1
            return { 'type': 'variable', 'value': varname }
        
        raise Exception('Illegal symbol: ' + current)



class SyntaxTree:
    """
    The tree representation of a boolean formula that results from parsing it.

    Types: operator, const, variable
    """


    def __init__(self, type: TreeType, value, children=[]):
        self.type = type
        self.value = value
        self.children = children

    @staticmethod
    def __applyOperatorConstant(operator, const, targetTree):
        if operator == 'and':
            return targetTree if const else SyntaxTree(TreeType.CONSTANT, 0)
        elif operator == 'or':
            return SyntaxTree(TreeType.CONSTANT, 1) if const else targetTree
        elif operator == 'xor':
            # x ^ 1 = ~x
            # x ^ 0 = x
            return SyntaxTree(TreeType.OPERATOR, 'not', [targetTree]) if const else targetTree
        else:
            raise Exception('Invalid binary operator ' + operator)

    def removeConstants(self):
        if self.type == TreeType.OPERATOR:
            # Simplify children
            newChildren = []
            for c in self.children:
                newChildren.append(c.removeConstants())
            # self.children = newChildren

            if self.value == 'not':
                c = newChildren[0]
                if c.type == TreeType.CONSTANT:
                    # Invert child and set as own value
                    return SyntaxTree(TreeType.CONSTANT, not c.value)
                elif c.type == TreeType.OPERATOR and c.value == 'not':
                    # Two NOTs resolve to identity
                    c = c.children[0]
                    return SyntaxTree(c.type, c.value, c.children)
                # Cannot simplify
                return SyntaxTree(self.type, self.value, newChildren)
            
            # Now it's one of { and, or, xor}
            left = newChildren[0]
            right = newChildren[1]
            if left.type == TreeType.CONSTANT and right.type == TreeType.CONSTANT:
                # Two constants --> result is also const
                val = binaryOperatorFunctions[self.value](left.value, right.value)
                return SyntaxTree(TreeType.CONSTANT, val)
            elif left.type == TreeType.CONSTANT:
                return SyntaxTree.__applyOperatorConstant(self.value, left.value, right)
            elif right.type == TreeType.CONSTANT:
                return SyntaxTree.__applyOperatorConstant(self.value, right.value, left)
        # Nothing to simplify
        return self

    
    def eliminateXor(self):
        if self.type == TreeType.OPERATOR:
            if self.value == 'xor':
                # x ^ y = (-x & y) | (x & -y)
                x = self.children[0]
                y = self.children[1]
                x_ = SyntaxTree(TreeType.OPERATOR, 'not', [x])
                y_ = SyntaxTree(TreeType.OPERATOR, 'not', [y])
                z1 = SyntaxTree(TreeType.OPERATOR, 'and', [x_, y])
                z2 = SyntaxTree(TreeType.OPERATOR, 'and', [x, y_])
                return SyntaxTree(TreeType.OPERATOR, 'or', [z1, z2])
            else:
                return SyntaxTree(self.type, self.value, [c.eliminateXor() for c in self.children])
        return self

    
    def nnf(self):
        raise Exception('Not yet working')
        if self.type == TreeType.OPERATOR:
            if self.value == 'not':
                child = self.children[0]
                if child.type == TreeType.CONSTANT or child.type == 'variable':
                    return self
                # child.type == 'operator'
                if child.value == 'not':
                    # Two chained NOTs resolve each other
                    return child.children[0].nnf()
                # Apply DeMorgan's rule
                c1 = SyntaxTree('operator', 'not', [child.children[0]]).nnf()
                c2 = SyntaxTree('operator', 'not', [child.children[1]]).nnf()
                if child.value == 'and':
                    return SyntaxTree('operator', 'or', [c1, c2])
                elif child.value == 'or':
                    return SyntaxTree('operator', 'and', [c1, c2])
                else:
                    raise Exception('Illegal operator for NNF: ' + self.value + '. Make sure to eliminate all operators except AND, OR an NOT before calling nnf()')
            # Other operators: convert children to NNF
            return SyntaxTree(self.type, self.value, [c.nnf() for c in self.children])
        else: # No operator
            return self

    
    def cnf(self):
        raise Exception('Not yet working')
        if self.type == 'operator':
            if self.value == 'not':
                # Formula is already in NNF, so only atoms come inside of a NOT
                return self
            if self.value == 'and':
                # Make sure children are in CNF
                return SyntaxTree(self.type, self.value, [c.cnf() for c in self.children])
            if self.value == 'or':
                # Apply distributive law
                l = self.children[0].cnf()
                r = self.children[1].cnf()
                if l.type == 'operator' and l.value == 'and':
                    ll = l.children[0]
                    lr = l.children[1]
                    c1 = SyntaxTree('operator', 'or', [ll, r]).cnf()
                    c2 = SyntaxTree('operator', 'or', [lr, r]).cnf()
                    return SyntaxTree('operator', 'and', [c1, c2])
                elif r.type == 'operator' and r.value == 'and':
                    rl = r.children[0]
                    rr = r.children[1]
                    c1 = SyntaxTree('operator', 'or', [rl, l]).cnf()
                    c2 = SyntaxTree('operator', 'or', [rr, l]).cnf()
                    return SyntaxTree('operator', 'and', [c1, c2])
                return SyntaxTree('operator', 'or', [l, r])
            else:
                raise Exception('Illegal operator for CNF: ' + self.value + '. Make sure the tree is in NNF before calling cnf()')
        else: # No operator
            return self


    def fill(self, variables):
        if self.type == TreeType.CONSTANT:
            return self
        if self.type == 'variable':
            val = variables.get(self.value)
            if val == None:
                return self
            return SyntaxTree(TreeType.CONSTANT, bool(val))
        # type == 'operator'
        newChildren = [c.fill(variables) for c in self.children]
        return SyntaxTree(self.type, self.value, newChildren)        

    # def simplify(self):
    #     if self.type == TreeType.CONSTANT or self.type == 'variable':
    #         return self, str(self.value)
    #     # type == 'operator'
    #     s = ''
    #     for c in self.children:
    #         co, cs = c.simplify()
            

    # def equalsExactly(self, other):
    #     if self.type != other.type:
    #         return False
    #     if self.type == TreeType.CONSTANT or self.type == 'variable':
    #         return self.value == other.value
    #     # type == 'operator'



    def print(self, filler='  ', depth=0):
        print(filler*depth + str(self.value))
        for c in self.children:
            c.print(filler, depth + 1)
    
    def __str__(self) -> str:
        val = str(int(self.value) if self.type == TreeType.CONSTANT else self.value)
        if self.type == TreeType.OPERATOR:
            val = inverseOperatorDict[val]
        if len(self.children) == 0:
            return val
        elif len(self.children) == 1:
            if len(self.children[0].children) > 1:
                return f"{val}({self.children[0]})"    
            return f"{val}{self.children[0]}"
        else:
            cs = []
            for c in self.children:
                if len(c.children) > 1 and operatorPrecedence.index(c.value) < operatorPrecedence.index(self.value):
                    cs.append(f'({c})')
                else:
                    cs.append(str(c))
            return f"{cs[0]} {val} {cs[1]}"

    def getVariableNames(self):
        if self.type == TreeType.CONSTANT:
            return {}
        if self.type == TreeType.VARIABLE:
            return { self.value }
        res = set()
        for c in self.children:
            res = res.union(c.getVariableNames())
        return res



# Token types: operator, const, variable, (, ), EOF
class BooleanParser:
    def __init__(self, tokenSource):
        if isinstance(tokenSource, BooleanLexer):
            self.lexer = tokenSource
        else:
            self.lexer = BooleanLexer(tokenSource)
        self.syntax_stack = []
        self.syntax_tree = None


    def __process_token(self, token):
        t = token['type']
        if t == 'operator':
            if len(self.syntax_stack) == 0:
                raise SyntaxError('Syntax stack is empty! Expected operand!')
            children = [self.syntax_stack.pop()]
            op = token['value']
            if op != 'not':
                if len(self.syntax_stack) == 0:
                    raise SyntaxError('Syntax stack is empty! Expected second operand!')
                children.insert(0, self.syntax_stack.pop())
            self.syntax_stack.append(SyntaxTree(TreeType.OPERATOR, op, children))
        elif t == 'variable':
            self.syntax_stack.append(SyntaxTree(TreeType.VARIABLE, token['value']))
        elif t == 'const':
            self.syntax_stack.append(SyntaxTree(TreeType.CONSTANT, token['value']))
        else:
            raise SyntaxError(f'Token type {t} cannot be processed in syntax tree')
    

    def parse(self) -> SyntaxTree:
        stack = []
        # Shunting Yard algorithm from Wikipedia
        while self.lexer.hasNext():
            token = self.lexer.next()
            type = token['type']
            if type == 'const':
                val = token['value']
                if val < 0 or val > 1:
                    raise SyntaxError(f'Not a boolean value: {val}')
                self.__process_token(token)
            elif type == 'variable':
                self.__process_token(token)
            elif type == 'operator':
                op = token['value']
                if op == 'not':
                    stack.append(token) # Treat unary operator like variable
                else:
                    prec = operatorPrecedence.index(op)
                    while len(stack) > 0 and \
                            stack[-1]['type'] == 'operator' and \
                            prec <= operatorPrecedence.index(stack[-1]['value']):
                        self.__process_token(stack.pop())
                    stack.append(token)
            elif type == '(':
                stack.append(token)
            elif type == ')':
                while True:
                    if len(stack) == 0:
                        raise SyntaxError('Expected "("')
                    if stack[-1]['type'] == '(':
                        break
                    self.__process_token(stack.pop())
                stack.pop()
            else:
                raise SyntaxError('Unexpected ' + type)
        
        while len(stack) > 0:
            token = stack.pop()
            if token['type'] == '(':
                raise SyntaxError('More opening than closing brackets')
            self.__process_token(token)
        
        if len(self.syntax_stack) == 0:
            raise SyntaxError('Expression does not contain a boolean formula')
        if len(self.syntax_stack) > 1:
            raise SyntaxError('Expression did not resolve to a valid syntax tree: too many root nodes left, expected more operators')
        return self.syntax_stack[0]



class BooleanFormula:
    def __init__(self, formula):
        if isinstance(formula, SyntaxTree):
            self.tree = formula
        else:
            self.tree = BooleanParser(formula).parse()


    def simplify(self):
        return BooleanFormula(self.tree.removeConstants())

    def fill(self, variables):
        return BooleanFormula(self.tree.fill(variables))

    def getVariableNames(self):
        return self.tree.getVariableNames()

    def __str__(self):
        return str(self.tree)

    def isTrue(self):
        return self.tree.type == TreeType.CONSTANT and self.tree.value == True
    def isFalse(self):
        return self.tree.type == TreeType.CONSTANT and self.tree.value == False

    def print_tree(self):
        self.tree.print()


if __name__ == '__main__':
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument('mode', help='"lex" or "parse"')
    args.add_argument('expression', help="expression to lex/parse")
    args.add_argument('-s', '--simplify', action='store_true', help='Simplify expression (only for mode "parse")')
    args = args.parse_args()
    expr = args.expression

    lexer = BooleanLexer(expr)
    if args.mode == 'lex':
        while lexer.hasNext():
            print(lexer.next())
    elif args.mode == 'parse':
        parser = BooleanParser(expr)
        formula = BooleanFormula(parser.parse())
        if args.simplify:
            formula = formula.simplify()
        print(formula)
    else:
        print('Invalid mode: ' + args.mode)