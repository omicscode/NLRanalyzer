# NLRanalyzer

- rust NLR analyzer which fetches and mines resistance genes.
- aligns your resistance genes to in-hosted resistance genes from previously published studies.
- implements a deep learning model based on the Neural Network to filter the alignment having the same sites and then use the expression labels and prediction of the sequences based on the expression.So an expression to sequences classifier using the libtorch.
- python directory contains the python version.
- libtorch should be present.

```
cargo build
```

```
NLRanalyzer.
       ************************************************
       Author Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************

Usage: NLRanalyzer <COMMAND>

Commands:
  miner    NLRResistanceMiner
  fetcher  NLRResistanceFetcher
  help     Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

Gaurav Sablok
codeprog@icloud.com
