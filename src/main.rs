mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
mod locatenlr;
mod resistancealignment;
mod resistancefetcher;
mod resistancelinear;
mod resistanceminer;
use crate::locatenlr::repeat_locator_long_read;
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
        Commands::TensorReady { refseq, outputfile } => {
            let command = alignment(refseq, outputfile).unwrap();
            println!("The command has finished:{}", command);
        }
        Commands::NLRlocate {
            pathfasta,
            locatenlr,
        } => {
            let file_path = pathfasta;
            let nlrlocate = Some(locatenlr);

            match repeat_locator_long_read(file_path, poly_type) {
                Ok(records) => {
                    for record in records {
                        println!("ID: {}", record.id);
                        println!(
                            "Sequence: {}...",
                            &record.sequence[..50.min(record.sequence.len())]
                        );
                        println!("Repeats: {:?}", record.repeat_locator);
                        println!("Fraction Lengths: {:?}", record.fraction_length);
                        println!("Coverage: {}", record.fraction_length_coverage);
                        println!("---");
                    }
                }
                Err(e) => eprintln!("Error: {}", e),
            }
            Ok(())
        }
    }
}
