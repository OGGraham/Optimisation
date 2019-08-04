import networkx as nx
import itertools
from pulp import *
from fractions import Fraction


def fractional_clique_cover(G):
    # Create xs (init to 0) for all subsets of vertices
    xs = {}
    for L in range(0, len(G.nodes) + 1):
        for subset in itertools.combinations(G.nodes, L):
            xs[subset] = 0

    # Clique is a complete subgraph - K(G) denotes all cliques in G (includes empty set & singular nodes)
    # Calculate K(G) (all cliques)
    K = list(nx.enumerate_all_cliques(G))

    # Formulate as LP problem
    prob = LpProblem("Fraction Clique Cover", LpMinimize)

    # Problem Vars
    lp_xs = LpVariable.dicts("Xs", xs, lowBound=0, upBound=None, cat='Continuous')

    # Add objective function first
    prob += lpSum(lp_xs), "Sum of Xs"

    # Add constraints to problem
    # 1. Xs = 0 iff S not a clique -> redundant (as explained in report)

    # 2. For any vertex v, the sum of the weights of the cliques containing v is >= 1
    for vertex in G.nodes:
        cliques_containing_v = []
        for clique in K:
            if vertex in clique:
                cliques_containing_v.append(clique)
        # Cliques containing v now contains the indexes of all
        # cliques that contain the current vertex
        prob += lpSum(lp_xs[tuple(x)] for x in cliques_containing_v) >= 1  # 2.

    # Solve
    prob.solve()

    print("Fractional Clique Cover:")
    print("Status -", LpStatus[prob.status])
    total = 0
    for item in lp_xs:
        val = lp_xs[item].varValue
        total += val

    total = Fraction.from_float(total).limit_denominator(100)

    print("Sum of Optimal Xs Vals -", total)
    return total, lp_xs


def shannon_entropy(G):
    # Create xs (init to 0) for all subsets of vertices
    xs = {}
    for L in range(0, len(G.nodes) + 1):
        for subset in itertools.combinations(G.nodes, L):
            xs[subset] = 0

    # Formulate as LP problem
    prob = LpProblem("Shannon Entropy", LpMaximize)

    # Problem Var
    lp_xs = LpVariable.dicts("Xs", xs, lowBound=0, upBound=None, cat='Continuous')

    # Add objective function first
    prob += lpSum(lp_xs[tuple(list(xs.keys())[-1])]), "Value of Xv"

    # Add problem constraints
    # 1. X_empty_set = 0
    prob += lp_xs[tuple()] == 0

    # 2. X_v <= 1 i.e. for any subset with a singular node, its value <= 1
    for item in lp_xs:
        if len(item) == 1:
            prob += lp_xs[item] <= 1

    # 3. X_N(v)_union_v - X_N(v) = 0
    for vertex in G.nodes:
        N_v = list(G.neighbors(vertex))
        N_v_u_v = list(G.neighbors(vertex))
        N_v_u_v.append(vertex)
        # Sort so in key order (as otherwise throws key error)
        N_v.sort()
        N_v_u_v.sort()
        prob += lpSum(lp_xs[tuple(N_v_u_v)]-lp_xs[tuple(N_v)]) == 0

    # 4. X_T - X_S >= 0 where S is a subset of T
    # 5. X_S + X_T - X_S_union_T - X_S_intersect_T >= 0
    # If S is a subset of T, then S union T == T & S intersect T is S == S
    for x in range(len(xs)-2):
        S = list(xs)[x]
        for y in range(x+1, len(xs)-1):
            T = list(xs)[y]
            # If S is subset of T
            if set(S).issubset(T):
                prob += lpSum(lp_xs[T] - lp_xs[S]) >= 0  # 4.
            if not set(S).issubset(T):  # This constraint is redundant if S is a subset of T
                X_s_u_t = list(set(S + T))  # Union
                X_s_n_t = [val for val in S if val in T]  # Intersection
                X_s_u_t.sort()  # Sort to prevent key error
                X_s_n_t.sort()
                prob += lpSum(lp_xs[S] + lp_xs[T] - lp_xs[tuple(X_s_u_t)] - lp_xs[tuple(X_s_n_t)]) >= 0  # 5.

    # Solve problem
    prob.solve()

    print("Shannon Entropy: ")
    print("Status -", LpStatus[prob.status])
    val = lp_xs[tuple(list(lp_xs)[-1])].varValue

    val = Fraction.from_float(val).limit_denominator(100)
    print("Xv -", val)

    return val, lp_xs


def calculate(G):
    fcc, fcc_vals = fractional_clique_cover(G)
    shann_ent, shannon_vals = shannon_entropy(G)
    return fcc, shann_ent, fcc_vals, shannon_vals


def create_graph():
    # Create graph
    G = nx.Graph()

    nodes = 0
    while nodes <= 1:
        try:
            nodes = int(input("How many nodes? "))
            if nodes == 1:
                raise ValueError
        except ValueError:
            print("Invalid Entry.")
    G.add_nodes_from([x for x in range(nodes)])

    cont = True
    while cont:
        print("Please enter two nodes to create an edge between:")
        print("Nodes (starts @ 0):", G.number_of_nodes(), "| Edges:", G.edges())
        try:
            n1 = int(input("Node 1 - "))
            n2 = int(input("Node 2 - "))
            if n1 != n2 and n1 in range(nodes) and n2 in range(nodes):
                G.add_edge(n1, n2)
                # Check if still want to add more nodes
                choice = input("Add another edge? (Y or N): ")
                if choice == "N" or choice == "n":
                    break
            else:
                print("Invalid Entry")
        except ValueError:
            print("Invalid Entry")
    return G


def load(filename):
    # Graphs stored in file format
    # line 1 - total number of nodes
    # line 2 - edges u1, v1;u2, v2; etc ...
    try:
        # Create new graph
        G = nx.Graph()
        with open(filename, 'r') as file:
            line = file.readline()
            # First line is number of nodes in graph
            G.add_nodes_from([x for x in range(int(line))])
            # Second Line is list of edges
            edges = file.readline().split(";")
            for edge in edges:
                G.add_edge(int(edge.split(",")[0]), int(edge.split(",")[1]))
        print("Graph successfully loaded.")
        return G
    except IOError:
        print("Loading failed - file does not exist.")
    except Exception:
        print("Something went wrong... likely invalid file format?")


def save(fcc, shannon_ent, fcc_vals, shannon_vals, G):
    if fcc == -1 or shannon_ent == -1:
        print("No results to save. (Please run calculate before attempting to save.)")
        return
    # Save fcc & shannon_ent in file w inputted filename
    filename = input("Name of file to be saved: ")
    with open(filename + ".txt", 'w+') as file:
        file.write("Graph: \n")
        file.write("Number Of Nodes: " + str(G.number_of_nodes()))
        file.write("\nEdges: " + str(G.edges))
        file.write("\n\nResults:")
        file.write("\nFractional Clique Cover - " + str(fcc))
        file.write("\nShannon Entropy - " + str(shannon_ent))
        file.write("\n\nFractional Clique Cover (All Values):\n")
        for item in fcc_vals:
            file.write(str(item) + " - " + str(Fraction.from_float(fcc_vals[item].varValue).limit_denominator(100)) + "\n")
        file.write("\n\nShannon Entropy (All Values):\n")
        for item in shannon_vals:
            file.write(str(item) + " - " + str(Fraction.from_float(shannon_vals[item].varValue).limit_denominator(100)) + "\n")

    print("Saving of " + filename + ".txt successful.")


def main():
    # Graph Vars
    G = nx.Graph()
    fcc, shannon_ent = -1, -1
    fcc_vals, shannon_vals = [], []
    while True:
        x = -1
        while x not in [y for y in range(0, 5)]:
            try:
                print("Optimisation Summative: \n 0 - Exit \n 1 - New Graph \n 2 - Load Graph \n 3 - Calculate \n 4 - Save result")
                x = int(input("Please choose an action: ").strip())
                if x not in [y for y in range(0, 5)]:
                    print("Choice outside range.")
            except ValueError:
                print("Invalid Entry.")

        # Carry Out Action Chosen
        if x == 0:
            # Exit
            break
        elif x == 1:
            # New graph
            G = create_graph()
            fcc, shannon_ent = -1, -1
            fcc_vals, shannon_vals = [], []
        elif x == 2:
            # Load Graph
            filename = input("Filename: ")
            G = load(filename)
            fcc, shannon_ent = -1, -1
            fcc_vals, shannon_vals = [], []
        elif x == 3:
            # Calculate
            fcc, shannon_ent, fcc_vals, shannon_vals = calculate(G)
        elif x == 4:
            # Save result
            save(fcc, shannon_ent, fcc_vals, shannon_vals, G)


# Call main loop
main()
