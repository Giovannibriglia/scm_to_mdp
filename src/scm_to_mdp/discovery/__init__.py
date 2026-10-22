"""Step S2: data -> Candidates (a class of 2-slice graphs, or scored parent sets). Each module has
fit(dataset, ...) -> Candidates; step S3 (scm_to_mdp.selection) then chooses one Graph.

The heavy libraries (Tigramite, pgmpy) are imported only by the module that uses them:
    from scm_to_mdp.discovery import pcmci_plus
"""
