rule parameter_space:
    output:
        "src/data/parameters.csv"
    cache:
        True
    script:
        "src/scripts/ic.py"

# ── Data unpacked from Zenodo Sandbox tarballs ────────────────────────────────

rule unpack_pluto:
    input:
        "src/data/pluto.tar.gz"
    output:
        directory("src/data/PLUTO_runs"),
    shell:
        "tar xzf {input} -C src/data/"

rule unpack_pop_spock:
    input:
        "src/data/pop_spock.tar.gz"
    output:
        directory("src/data/pop_compact"),
        directory("src/data/pop_compact_reduced"),
        directory("src/data/pop_spreading"),
        directory("src/data/pop_spreading_extern"),
        directory("src/data/pop_spreading_norcrit"),
        directory("src/data/pop_spreading_reduced"),
        directory("src/data/pop_spreading_reduced_extern"),
        directory("src/data/pop_spreading_reduced_norcrit"),
    shell:
        "tar xzf {input} -C src/data/"

rule unpack_single_spock:
    input:
        "src/data/single_spock.tar.gz"
    output:
        directory("src/data/single_internal_5au"),
        directory("src/data/single_internal_5au_factor10"),
        directory("src/data/single_internal_5au_factor10_external_G01"),
        directory("src/data/single_internal_5au_factor10_external_G010"),
        directory("src/data/single_internal_5au_factor10_external_G0100"),
        directory("src/data/single_internal_5au_factor10_norcrit"),
        directory("src/data/single_internal_5au_norcrit"),
        directory("src/data/single_internal_20au"),
        directory("src/data/single_internal_20au_factor10"),
        directory("src/data/single_internal_20au_factor10_external_G01"),
        directory("src/data/single_internal_20au_factor10_external_G010"),
        directory("src/data/single_internal_20au_factor10_external_G0100"),
        directory("src/data/single_internal_20au_factor10_norcrit"),
        directory("src/data/single_internal_20au_norcrit"),
        directory("src/data/single_internal_80au"),
        directory("src/data/single_internal_80au_factor10"),
        directory("src/data/single_internal_80au_factor10_external_G01"),
        directory("src/data/single_internal_80au_factor10_external_G010"),
        directory("src/data/single_internal_80au_factor10_external_G0100"),
        directory("src/data/single_internal_80au_factor10_norcrit"),
        directory("src/data/single_internal_80au_norcrit"),
    shell:
        "tar xzf {input} -C src/data/"
