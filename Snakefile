rule parameter_space:
    output:
        "src/data/parameters.csv"
    script:
        "src/scripts/initial_conditions.py"
