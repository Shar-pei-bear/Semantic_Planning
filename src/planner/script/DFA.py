
class DFA:

    """This is a deterministic Finite State Automaton (NFA).
    """

    def __init__(self, initial_state=None, alphabet=None, transitions= dict([]),final_states=set(), goal_states={}, memory=None):
        self.state_transitions = {}
        self.final_states = final_states
        self.goal_states = goal_states
        self.state_transitions=transitions
        if alphabet == None:
            self.alphabet=[]
        else:
            self.alphabet=alphabet
        self.initial_state = initial_state
        self.states =[ initial_state ] # the list of states in the machine.

    def reset (self):

        """This sets the current_state to the initial_state and sets
        input_symbol to None. The initial state was set by the constructor
         __init__(). """

        self.current_state = self.initial_state
        self.input_symbol = None

    def add_transition(self,input_symbol,state,next_state=None):
        if next_state is None:
            next_state = state
        else:
            self.state_transitions[(input_symbol, state)] = next_state
        if next_state in self.states:
            pass
        else:
            self.states.append(next_state)
        if state in self.states:
            pass
        else:
            self.states.append(state)

        if input_symbol in self.alphabet:
            pass
        else:
            self.alphabet.append(input_symbol)


    def get_transition(self, input_symbol, state):
        """This returns a list of next states given an input_symbol and state.
        """

        if self.state_transitions.has_key((input_symbol, state)):
            return self.state_transitions[(input_symbol, state)]
        else:
            return None

if __name__=='__main__':
    #construct a DRA, which is a complete automaton.
    dra=DFA(0,['1','2','3','4','E']) # we use 'E' to stand for everything else other than 1,2,3,4.
    # dra.add_transition('2',0,0)
    # dra.add_transition('3',0,0)
    # dra.add_transition('E',0,0)
    #
    # dra.add_transition('1',0,1)
    # dra.add_transition('1',1,2)
    # dra.add_transition('3',1,2)
    # dra.add_transition('E',1,2)
    #
    # dra.add_transition('2',1,3)
    #
    # dra.add_transition('1',2,2)
    # dra.add_transition('3',2,2)
    # dra.add_transition('E',2,2)
    #
    # dra.add_transition('2',2,3)
    #
    # dra.add_transition('1',3,3)
    # dra.add_transition('2',3,3)
    # dra.add_transition('E',3,3)
    #
    # dra.add_transition('3',3,0)
    #
    # dra.add_transition('4',0,4)
    # dra.add_transition('4',1,4)
    # dra.add_transition('4',2,4)
    # dra.add_transition('4',3,4)
    # dra.add_transition('4',4,4)
    # dra.add_transition('1',4,4)
    # dra.add_transition('2',4,4)
    # dra.add_transition('3',4,4)
    # dra.add_transition('E',4,4)
    #
    # J0={4}
    # K0={1}
    # rabin_acc=[(J0,K0)]
    # dra.add_rabin_acc(rabin_acc)


