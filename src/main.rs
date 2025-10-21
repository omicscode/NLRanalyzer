mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
mod resistancealignment;
mod resistancefetcher;
mod resistancelinear;
mod resistanceminer;
use crate::resistancealignment::alignment;
use crate::resistancefetcher::prgdb_sequence_fetcher;
use crate::resistanceminer::mine_resistance_genes;
/*
Author Gaurav Sablok,
Email: codeprog@icloud.com
*/

fn main() {
    let argparse = CommandParse::parse();
    match &argparse.command {
        Commands::Fetcher { idstring, sequence } => {
            let command = prgdb_sequence_fetcher(idstring, sequence).unwrap();
            println!("The command has finished:{}", command);
        }
        Commands::Miner { idstring } => {
            let genbank_id = mine_resistance_genes(idstring).unwrap();
            println!("Resistance gene GenBank ID: {}", genbank_id);
        }
        Commands::TensorReady { refseq } => {
            let command = alignment(refseq).unwrap();
            println!("The command has finished:{}", command);
        }
    }
}
