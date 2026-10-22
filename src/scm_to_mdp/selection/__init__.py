"""Step S3: from the candidates of a discovery method (Candidates) to ONE graph (Graph), by a
rule fixed before seeing the data. Each module has select(candidates, dataset) -> Graph.

    bic_best   the candidate with the best BIC on the data (ties: the best transition BIC)
    given      the candidates are already one graph (the oracle): take it
"""
