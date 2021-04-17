use nalgebra::base::DMatrix;
use std::path::Path;
use std::fs::File;

fn main() {
    println!("Decoding image");
    let decoder = png::Decoder::new(File::open("res/cows.png").unwrap());
    let (info, mut reader) = decoder.read_info().unwrap();
    println!("{:?}", info);

    println!("Creating buffer");
    let mut buf = vec![0; info.buffer_size()];

    println!("Reading frame");
    reader.next_frame(&mut buf).unwrap();

    println!("Creating matrix");
    let mut mat = DMatrix::from_element(info.width as usize, info.height as usize, [0, 0, 0]);
    let mut pixels = buf.chunks_exact(3);
    for y in 0..info.height {
        for x in 0..info.width {
            let pixels = pixels.next().unwrap();
            mat[(x as usize, y as usize)] = [pixels[0], pixels[1], pixels[2]];
        }
    }

    println!("Creating output file");
    let path = Path::new("out.png");
    let file = File::create(path).unwrap();
    let w = std::io::BufWriter::new(file);

    println!("Preparing encoder");
    let mut encoder = png::Encoder::new(w, info.width, info.height);
    encoder.set_color(info.color_type);
    encoder.set_depth(info.bit_depth);

    println!("Encoding and saving image");
    let mut writer = encoder.write_header().unwrap();
    let mut data = Vec::new();
    for pixel in mat.into_iter() {
        data.extend(pixel.iter());
    }
    writer.write_image_data(&data).unwrap();
}
