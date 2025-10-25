use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::Command;
use std::thread;
use tch::Tensor;

/*
Author Gaurav Sablok,
Email: codeprog@icloud.com
*/

#[derive(Debug, Clone, PartialOrd, PartialEq)]

pub struct Mapper {
    pub id: String,
    pub seq: Vec<Tensor>,
}

#[tokio::main]
pub async fn alignment(pathref: &str, outputfilename: &str) -> Result<String, Box<dyn Error>> {
    let _ = Command::new("mafft")
        .arg("-")
        .arg("lnsi")
        .arg(pathref)
        .arg(">")
        .arg(outputfile)
        .output()
        .expect("file not present");

    let mut pathread: Vec<String> = Vec::new();
    let mut pathid: Vec<String> = Vec::new();

    let fileopen = File::open(outputfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    for i in fileread.lines() {
        let line = i.expect("line not present");
        if line.starts_with(">") {
            let newline = line.replace(">", "");
            pathid.push(newline);
        } else {
            pathread.push(line);
        }
    }

    let finalvec: Vec<String> = Vec::new();
    let newvecid: Vec<Vec<String>> = Vec::new();

    // an assertion check
    let lengthcheck = pathread.into_iter().map(|x| x.len()).collect::<Vec<_>>();
    let a = lengthcheck.iter().sum() as f32;
    let b = lengthcheck.iter().sum() as f32;
    let c = a as f32 / b as f32 as f32;
    assert_eq!(a, b);

    // filtering the similar sites
    let mut filteredseq: Vec<String> = Vec::new();
    thread::spawn(move || {
        for i in 0..pathread.len() - 1 {
            let seqhold = pathread[i].clone();
            let seqholdnext = pathread[i + 1].clone();
            let mut seqfilter: Vec<String> = Vec::new();
            for i in seqhold.chars().map(|x| String::from(x)) {
                for j in seqholdnext.chars().map(|x| String::from(x)) {
                    if i == j {
                        continue;
                    } else if i != j {
                        seqfilter.push(i.clone());
                        filteredseq.push(seqfilter.join("").to_string());
                    }
                }
            }
            finalvec.push(pathid[i]);
        }
    });

    let finaltensorvec: Vec<Mapper> = Vec::new();
    thread::spawn(move || {
        for i in 0..filteredseq.len() {
            let valuetensor = encodenlr(&filteredseq[i]);
            let valueid = finalvec[i];
            finaltensorvec.push(Mapper {
                id: valueid,
                seq: valuetensor,
            });
        }
    });

    Ok("The value has been written".to_string())
}

// creating the tensor
fn encodenlr(seq: &str) -> Vec<Tensor> {
    let mapping: HashMap<char, [f32; 4]> = [
        ('A', [1.0, 0.0, 0.0, 0.0]),
        ('T', [0.0, 1.0, 0.0, 0.0]),
        ('C', [0.0, 0.0, 1.0, 0.0]),
        ('G', [0.0, 0.0, 0.0, 1.0]),
    ]
    .iter()
    .cloned()
    .collect();
    let mut data = vec![0f32; seq.len() * 4];
    for (i, c) in seq.chars().enumerate() {
        let start = i * 4;
        if let Some(oh) = mapping.get(&c) {
            data[start..start + 4].copy_from_slice(oh);
        }
    }
    let mut tensorvec: Vec<Tensor> = Vec::new();
    // [batch=1, channels=4, length]
    tensorvec.push(Tensor::from_slice(&data).view([1, 4, seq.len() as i64]));
    return tensorVec;
}
