use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "NLRanalyzer",
    version = "1.0",
    about = "NLRanalyzer.
       ************************************************
       Author Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// NLRResistanceMiner
    Miner {
        /// id for the resistance gene
        idstring: String,
    },
    /// NLRResistanceFetcher
    Fetcher {
        /// id for the resistance gene
        idstring: String,
        /// dnasequence or protein sequence
        sequence: String,
    },
    /// align the sequences and prepare them for the tensor
    TensorReady {
        /// path to the reference sequence
        refseq: String,
        /// path to the output file
        outputfile: String,
    },
    /// locate NLR
    NLRlocate {
        /// path to the fasta file
        pathfasta: String,
        /// NLR motif to locate
        locatenlr: String,
    },
    /// train NLR
    TrainNLR {
        /// path to the fasta file
        fastafile: String,
        ///
        outputfile: String,
        /// expression file
        expressionfile: String,
    },
}
