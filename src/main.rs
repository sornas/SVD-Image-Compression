use log::*;
use peroxide::fuga::LinearAlgebra;
use std::path::Path;
use std::fs::File;
use std::time::Instant;

fn main() {
    setup_logger();

    debug!("Decoding image");
    let decoder = png::Decoder::new(File::open("res/cows800.png").unwrap());
    let (info, mut reader) = decoder.read_info().unwrap();
    debug!("{:?}", info);
    let width = info.width as usize;
    let height = info.height as usize;

    debug!("Creating buffer");
    let mut buf = vec![0; info.buffer_size()];

    debug!("Reading frame");
    reader.next_frame(&mut buf).unwrap();

    debug!("Extracting channels");
    let mut channels = [
        Vec::with_capacity(width * height),
        Vec::with_capacity(width * height),
        Vec::with_capacity(width * height),
    ];
    let mut pixels = buf.chunks_exact(3);
    for _p in 0..width * height {
        let pixels = pixels.next().unwrap();
        for i in 0..3 {
            channels[i].push(pixels[i] as f64);
        }
    }

    let mats = channels.iter().map(|m| {
        peroxide::fuga::matrix(m.clone(), width, height, peroxide::fuga::Row)
    }).collect::<Vec<_>>();

    debug!("Factorizing matrixes");
    let svds: Vec<_> = mats.iter().enumerate().map(|(i, m)| {
        debug!("Factorizing {}", i);
        let start = Instant::now();
        let svd = m.svd();
        let end = Instant::now();
        info!("Factorized in {:?}", end - start);
        svd
    }).collect();

    let rank = 500;

    debug!("Compressing matrixes");
    let compressed: Vec<Vec<f64>> = svds.into_iter().enumerate().map(|(i, m)| {
        debug!("Compressing {}", i);

        let u = {
            let mut u = Vec::with_capacity(m.u.row * rank);
            for y in 0..m.u.row {
                for x in 0..rank {
                    u.push(m.u[(x, y)]);
                }
            }
            peroxide::fuga::matrix(u, m.u.row, rank, peroxide::fuga::Row)
        };
        
        let s = {
            let mut s = vec![0.0; m.u.row * m.vt.col];
            for i in 0..m.s.len() {
                s[i * m.s.len() + i] = m.s[i];
            }
            peroxide::fuga::matrix(s, m.u.row, m.vt.col, peroxide::fuga::Row)
        };

        let v = {
            let mut v = Vec::with_capacity(rank * m.vt.col);
            for y in 0..rank {
                for x in 0..m.vt.col {
                    v.push(m.vt[(x, y)]);
                }
            }
            peroxide::fuga::matrix(v, rank, m.vt.col, peroxide::fuga::Row)
        };

        let start = Instant::now();
        let compressed = u * s * v;
        let end = Instant::now();
        info!("Compressed in {:?}", end - start);
        compressed.into()
    }).collect();

    debug!("Merging compressed matrixes");
    let mut data = Vec::new();
    for i in 0..compressed[0].len() {
        data.push(compressed[0][i].round() as u8);
        data.push(compressed[1][i].round() as u8);
        data.push(compressed[2][i].round() as u8);
    }

    debug!("Creating output file");
    let path = Path::new("out.png");
    let file = File::create(path).unwrap();
    let w = std::io::BufWriter::new(file);

    debug!("Preparing encoder");
    let mut encoder = png::Encoder::new(w, info.width, info.height);
    encoder.set_color(info.color_type);
    encoder.set_depth(info.bit_depth);

    debug!("Encoding and saving image");
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&data).unwrap();
}

fn setup_logger() {
    use colored::*;

    fern::Dispatch::new()
        .format(move |out, message, record| {
            let message = message.to_string();
            out.finish(format_args!(
                "{} {} {}:{}{}{}",
                match record.level() {
                    Level::Error => "ERROR".red(),
                    Level::Warn => "WARN ".yellow(),
                    Level::Info => "INFO ".normal(),
                    Level::Debug => "DEBUG".green(),
                    Level::Trace => "TRACE".normal(),
                },
                // chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S%.6f]"),
                chrono::Local::now().format("[%H:%M:%S%.6f]"),
                record.file().unwrap(),
                record.line().unwrap(),
                if message.chars().any(|e| e == '\n') {
                    "\n"
                } else {
                    " "
                },
                message
            ))
        })
        .level(LevelFilter::Debug)
        .chain(std::io::stderr())
        .apply()
        .unwrap();
}
