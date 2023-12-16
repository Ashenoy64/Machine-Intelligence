import torch
class HMM:
    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        self.states_dict = {state: i for i, state in enumerate(self.states)}
        self.emissions_dict = {emission: i for i, emission in enumerate(self.emissions)}

    def viterbi_algorithm(self, seq):
        T = len(seq)
        delta = [[0.0 for _ in range(self.N)] for _ in range(T)]
        psi = [[0 for _ in range(self.N)] for _ in range(T)]

        # Initialization
        for i in range(self.N):
            delta[0][i] = self.pi[i] * self.B[i][self.emissions_dict[seq[0]]]
            psi[0][i] = 0

        # Recursion
        for t in range(1, T):
            for j in range(self.N):
                max_prob = float('-inf')
                max_index = 0
                for i in range(self.N):
                    prob = delta[t - 1][i] * self.A[i][j]
                    if prob > max_prob:
                        max_prob = prob
                        max_index = i
                delta[t][j] = max_prob * self.B[j][self.emissions_dict[seq[t]]]
                psi[t][j] = max_index

        # Backtracking
        path = [0] * T
        path[T - 1] = max(range(self.N), key=lambda x: delta[T - 1][x])

        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]

        return [self.states[state] for state in path]
