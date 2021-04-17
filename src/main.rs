use log::*;
use nalgebra::base::{DMatrix, Matrix};
use nalgebra::linalg::SVD;
use std::path::Path;
use std::fs::File;

fn main() {
    setup_logger();

    info!("Decoding image");
    let decoder = png::Decoder::new(File::open("res/cows800.png").unwrap());
    let (info, mut reader) = decoder.read_info().unwrap();
    debug!("{:?}", info);

    info!("Creating buffer");
    let mut buf = vec![0; info.buffer_size()];

    info!("Reading frame");
    reader.next_frame(&mut buf).unwrap();

    info!("Creating matrixes");
    let mut mats = [
        DMatrix::from_element(info.width as usize, info.height as usize, 0.0),
        DMatrix::from_element(info.width as usize, info.height as usize, 0.0),
        DMatrix::from_element(info.width as usize, info.height as usize, 0.0),
    ];
    let mut pixels = buf.chunks_exact(3);
    for y in 0..info.height {
        for x in 0..info.width {
            let pixels = pixels.next().unwrap();
            for i in 0..3 {
                mats[i][(x as usize, y as usize)] = pixels[i] as f64;
            }
        }
    }

    info!("Factorizing matrixes");
    let svds: Vec<_> = mats.iter().enumerate().map(|(i, m)| {
        debug!("Factorizing {}", i);
        SVD::new(m.clone(), true, true)
    }).collect();

    let rank = 100;

    info!("Compressing matrixes");
    let compressed: Vec<_> = svds.into_iter().enumerate().map(|(i, mut m)| {
        debug!("Compressing {}", i);

        let u = m.u.take().map(|u| {
            let r = u.nrows();
            u.resize(r, rank, 0.0)
        }).unwrap();
        
        let sigma = Matrix::from_diagonal(&m.singular_values.resize_vertically(rank, 0.0));

        let v = m.v_t.take().map(|v| {
            let c = v.ncols();
            v.resize(rank, c, 0.0)
        }).unwrap();

        u * sigma * v
    }).collect();

    info!("Creating output file");
    let path = Path::new("out.png");
    let file = File::create(path).unwrap();
    let w = std::io::BufWriter::new(file);

    info!("Preparing encoder");
    let mut encoder = png::Encoder::new(w, info.width, info.height);
    encoder.set_color(info.color_type);
    encoder.set_depth(info.bit_depth);

    info!("Encoding and saving image");
    let mut writer = encoder.write_header().unwrap();
    let mut data = Vec::new();
    // for pixel in mat.into_iter() {
    //     data.extend(pixel.iter());
    // }
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
