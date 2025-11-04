use rand::Rng;
use std::collections::HashMap;
use tch::nn::{Module, Optimizer, VarList};
use tch::optim::Adam;
use tch::{Device, Kind, Tensor, nn, no_grad};
use tokio::sync::RwLockMappedWriteGuard;

/*
Author Gaurav Sablok,
Email: codeprog@icloud.com
*/

#[derive(Debug, Clone, PartialOrd, PartialEq)]

pub struct Mapper {
    pub id: String,
    pub seq: Vec<Tensor>,
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct MapperFinalLabels {
    pub seq: Vec<Tensor>,
    pub expression: f64,
}

#[tokio::main]
pub async fn deeplearnalignment(
    pathref: &str,
    outputfilename: &str,
    expressionfilename: &str,
) -> Result<String, Box<dyn Error>> {
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

    // labelling the alignment according to the associated expression.
    let mut filexpressionvec: Vec<MapperFinal> = Vec::new();
    let filexpressionopen = File::open(expressionfilename).expect("file not present");
    let fileexpressionread = Bufreader::new(filexpressionopen);
    for i in finaltensorvec.iter() {
        for val in fileexpressionread.lines() {
            let line = val.expect("file not found");
            let linevec = val.split("\t").collect::<Vec<_>>();
            if i.id == linevec[0].clone().to_string() {
                filexpressionvec.push(MapperFinal {
                    seq: i.seq,
                    expression: linevec[1].clone(),
                })
            }
        }
    }

    let device = Device::Cpu;
    let n_samples = 1000;
    let seq_length = 100;

    let seq: Vec<String> = finalexpressionvec
        .into_iter()
        .map(|x| x.seq)
        .collect::<Vec<_>>();

    let labels: Vec<usize> = finalexpressionvec
        .into_iter()
        .map(|x| x.expression as f64)
        .collect::<Vec<_>>();

    let x = Tensor::cat(&seq, 0).to_kind(Kind::Float);

    let y = Tensor::of_slice(&labels)
        .to_kind(Kind::Int64)
        .to_device(device);

    // Split
    let indices = Tensor::range(0, n_samples as i64, 1).shuffle_();
    let train_size = (n_samples as f32 * 0.8) as i64;
    let train_idx = indices.narrow(0, 0, train_size);
    let test_idx = indices.narrow(0, train_size, n_samples as i64 - train_size);

    let x_train = x.index_select(0, &train_idx);
    let y_train = y.index_select(0, &train_idx);
    let x_test = x.index_select(0, &test_idx);
    let y_test = y.index_select(0, &test_idx);

    // Model: CNN
    let vs = nn::VarStore::new(device);
    let conv1 = nn::conv1d(&vs.root(), 4, 32, 8, Default::default());
    let conv2 = nn::conv1d(&vs.root(), 32, 64, 8, Default::default());
    let linear1 = nn::linear(
        &vs.root(),
        64 * ((seq_length as i64 - 15) / 4),
        64,
        Default::default(),
    );
    let linear2 = nn::linear(&vs.root(), 64, 1, Default::default());

    let mut optim = Adam::default()
        .weight_decay(0.01)
        .build(&vs.trainable_variables(), 0.001)?; // L2 regularization

    // Training
    for epoch in 0..50 {
        let out = conv1.forward(&x_train).relu().max_pool1d(2, 2, 0, 1, false);
        let out = conv2.forward(&out).relu().max_pool1d(2, 2, 0, 1, false);
        let out = out.view([-1, 64 * ((seq_length as i64 - 15) / 4)]);
        let out = linear1.forward(&out).relu();
        let logits = linear2.forward(&out);
        let loss = logits.binary_cross_entropy_with_logits(
            &y_train.view([-1, 1]).to_kind(Kind::Float),
            None,
            None,
            tch::Reduction::Mean,
        );
        optim.backward_step(&loss);

        if epoch % 10 == 0 {
            println!("Epoch {} Loss: {:.4}", epoch, f32::from(loss));
        }

        // Early stopping
        let val_loss = no_grad(|| {
            let out = conv1.forward(&x_test).relu().max_pool1d(2, 2, 0, 1, false);
            let out = conv2.forward(&out).relu().max_pool1d(2, 2, 0, 1, false);
            let out = out.view([-1, 64 * ((seq_length as i64 - 15) / 4)]);
            let out = linear1.forward(&out).relu();
            let logits = linear2.forward(&out);
            logits.binary_cross_entropy_with_logits(
                &y_test.view([-1, 1]).to_kind(Kind::Float),
                None,
                None,
                tch::Reduction::Mean,
            )
        });
        if f32::from(val_loss) < 0.1 {
            println!("Early stopping at epoch {}", epoch);
            break;
        }
    }

    // Evaluate
    let pred = no_grad(|| {
        let out = conv1.forward(&x_test).relu().max_pool1d(2, 2, 0, 1, false);
        let out = conv2.forward(&out).relu().max_pool1d(2, 2, 0, 1, false);
        let out = out.view([-1, 64 * ((seq_length as i64 - 15) / 4)]);
        let out = linear1.forward(&out).relu();
        let logits = linear2.forward(&out).sigmoid();
        logits.gt(0.5).to_kind(Kind::Int64)
    });
    let accuracy = pred
        .eq(&y_test.view([-1, 1]))
        .to_kind(Kind::Float)
        .mean(Kind::Float);
    println!("Test Accuracy: {:.4}", f32::from(accuracy));

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
