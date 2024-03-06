import itertools

'''
    Part of Project AxoLoTL
    By Jackson Thissell
'''

'''
    Context-Free Grammar for essential boolean logic.
    Includes And, Or, Not, Equals, True and False; and A-G as variables.
'''

class Token:
    def __init__(self, symbol: str, embed: int):
        self.symbol = symbol
        self.embed = embed
    
# internal tokens
Token.BEGIN = Token("◆", 0)
Token.ENTAILS = Token("⊢", 1)
Token.MASK = Token("?", 2)

# constants
Token.TRUE = Token("1", 3)
Token.FALSE = Token("0", 4)

# operators
Token.AND = Token('·', 5)
Token.OR = Token('+', 6)
Token.NOT = Token('¬', 7)
Token.EQUALS = Token("=", 8)

# variables
Token.VAR_A = Token("A", 9)
Token.VAR_B = Token("B", 10)
Token.VAR_C = Token("C", 11)
Token.VAR_D = Token("D", 12)
Token.VAR_E = Token("E", 13)
Token.VAR_F = Token("F", 14)
Token.VAR_G = Token("G", 15)

id_to_str_dict = {
    0: '◆',
    1: '⊢',
    2: '?',
    3: '1',
    4: '0',
    5: '·',
    6: '+',
    7: '¬',
    8: '=',
    9: 'A',
    10: 'B',
    11: 'C',
    12: 'D',
    13: 'E',
    14: 'F',
    15: 'G'
}

Domain = bool
LookupDict = dict[Token, Domain]

class Node:
    def __init__(self, token: Token):
        self.token = token

    def varset(self) -> set[Token]:
        pass

    def eval(self, _: None) -> Domain:
        pass


class ConstNode(Node):
    def __init__(self, token: Token, val: Domain):
        super(ConstNode, self).__init__(token)

        self.val = val

    def varset(self) -> set[Token]:
        return set()
    
    def eval(self, _: None) -> Domain:
        return self.val


class VarNode(Node):
    def __init__(self, token: Token):
        super(VarNode, self).__init__(token)

    def varset(self) -> set[Token]:
        return set([self.token])
    
    def eval(self, lookup_dict: None) -> Domain:
        return lookup_dict[self.token]


class UnaryNode(Node):
    def __init__(self, token: Token, child: Node):
        super(UnaryNode, self).__init__(token)

        self.child = child

    def varset(self) -> set[Token]:
        return self.child.varset()

    def eval(self, _: None) -> Domain:
        pass


class BinaryNode(Node):
    def __init__(self, token: Token, left: Node, right: Node):
        super(BinaryNode, self).__init__(token)

        self.left = left
        self.right = right

    def varset(self) -> set[Token]:
        return self.left.varset().union(self.right.varset())

    def eval(self, _: None) -> Domain:
        pass

# Individual Node definitions
    
class TrueNode(ConstNode):
    def __init__(self):
        super(TrueNode, self).__init__(Token.TRUE, True)

class FalseNode(ConstNode):
    def __init__(self):
        super(FalseNode, self).__init__(Token.FALSE, False)

class VarANode(VarNode):
    def __init__(self):
        super(VarANode, self).__init__(Token.VAR_A)

class VarBNode(VarNode):
    def __init__(self):
        super(VarBNode, self).__init__(Token.VAR_B)

class VarCNode(VarNode):
    def __init__(self):
        super(VarCNode, self).__init__(Token.VAR_C)

class VarDNode(VarNode):
    def __init__(self):
        super(VarDNode, self).__init__(Token.VAR_D)

class VarENode(VarNode):
    def __init__(self):
        super(VarENode, self).__init__(Token.VAR_E)

class VarFNode(VarNode):
    def __init__(self):
        super(VarFNode, self).__init__(Token.VAR_F)

class VarGNode(VarNode):
    def __init__(self):
        super(VarGNode, self).__init__(Token.VAR_G)

class NotNode(UnaryNode):
    def __init__(self, child: Node):
        super(NotNode, self).__init__(Token.NOT, child)
    
    def eval(self, lookup_dict) -> Domain:
        return not self.child.eval(lookup_dict)

class OrNode(BinaryNode):
    def __init__(self, left: Node, right: Node):
        super(OrNode, self).__init__(Token.OR, left, right)
    
    def eval(self, lookup_dict) -> Domain:
        return self.left.eval(lookup_dict) or self.right.eval(lookup_dict)
    
class AndNode(BinaryNode):
    def __init__(self, left: Node, right: Node):
        super(AndNode, self).__init__(Token.AND, left, right)

    def eval(self, lookup_dict) -> Domain:
        return self.left.eval(lookup_dict) and self.right.eval(lookup_dict)

class EqualsNode(BinaryNode):
    def __init__(self, left: Node, right: Node):
        super(EqualsNode, self).__init__(Token.EQUALS, left, right)

    def eval(self, lookup_dict) -> Domain:
        return self.left.eval(lookup_dict) == self.right.eval(lookup_dict)

class MaskNode(Node):
    def __init__(self):
        super(MaskNode, self).__init__(Token.MASK)

    def eval(self, _: None) -> Domain:
        raise Exception("wtf are you doing? how did a mask end up in evaluation?")

class EntailsNode(BinaryNode):
    def __init__(self, left: Node, right: Node):
        super(EntailsNode, self).__init__(Token.ENTAILS, left, right)

    def eval(self, _: None) -> Domain:
        varset = list(self.varset())
        pos_vals = [False, True]

        if len(varset) == 0:
            return not(self.left.eval(None)) or self.right.eval(None)
        else:
            p_set = itertools.product(pos_vals, repeat=len(varset))
            for p in p_set:
                lookup_dict = { i: v for i, v in zip(varset, p)}
                if self.left.eval(lookup_dict) and not(self.right.eval(lookup_dict)):
                    return False
            
            return True

class BeginNode(UnaryNode):
    def __init__(self, child):
        super(BeginNode, self).__init__(Token.BEGIN, child)

    def eval(self, _: None) -> Domain:
        return self.child.eval(None)


SAFE_NODE_SET = [EqualsNode, NotNode, AndNode, OrNode, TrueNode, FalseNode]
VAR_NODE_SET = [VarANode, VarBNode, VarCNode, VarDNode, VarENode, VarFNode, VarGNode]
TERM_NODE_SET = [TrueNode, FalseNode]