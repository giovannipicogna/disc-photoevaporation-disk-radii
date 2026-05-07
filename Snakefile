rule parameter_space:
    output:
        "src/data/parameters.csv"
    cache:
        True
    script:
        "src/scripts/ic.py"
