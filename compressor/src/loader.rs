use std::{fmt::{self, Display, Formatter}, path::PathBuf};

use image::GenericImageView;
use image::io::Reader as ImageReader;
use glob::glob;

#[derive(Clone, Debug)]
pub struct RawImageData {
    pub width: usize,
    pub height: usize,
    pub rows: Vec<Vec<u8>>,
    pub name: String,
    pub pixel_count: u64,
}

impl Display for RawImageData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}:\t{} rows, {} pixels", self.name, self.rows.len(), self.pixel_count)
    }
}

pub struct Loader {
    files: Vec<PathBuf>
}

impl Loader {
    pub fn create_from_filter(filter: &str) -> Loader {
        let files = glob(filter)
            .expect("Failed to read glob pattern")
            .filter_map(|item| item.ok())
            .collect();

        return Loader { files };
    }

    pub fn load_files(&self) -> Vec<RawImageData>
    {
        return self.files.iter()
            .map(|path| {
                let img = ImageReader::open(path)
                    .expect("Failed to open image")
                    .decode()
                    .expect("Failed to load image");

                let width = img.width() as usize;
                let height = img.height() as usize;
                let mut rows = Vec::with_capacity(height);
                let mut pixel_count = 0u64;

                for row in img.into_bgra8().chunks(width * 4) {
                    assert_eq!(row.len(), width * 4);

                    // Convert all pixels to 2x8 bit values
                    let row = row.chunks(4).map(pack_pixel).flatten().collect::<Vec<u8>>();
                    assert_eq!(row.len(), width * 2);

                    pixel_count += (row.len() / 2) as u64;
                    rows.push(row);
                }

                let name = sanitise_path(&path);

                RawImageData {
                    width,
                    height,
                    rows,
                    name,
                    pixel_count,
                }
            })
            .collect();
    }
}

fn sanitise_path(path: &PathBuf) -> String
{
    return path
        .to_string_lossy()
        .to_string()
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(".", "_");
}

fn pack_pixel(bgra: &[u8]) -> [u8; 2]
{
    assert_eq!(bgra.len(), 4);

    // return early if it's transparent
    if bgra[3] <= 0 {
        return [0, 0]
    }

    let r = bgra[2] as u16;
    let g = bgra[1] as u16;
    let b = bgra[0] as u16;

    // Pack to 16 bit and return 2 bytes
    let px = ((r >> 3) << 0) | ((g >> 3) << 6) | ((b >> 3) << 11) | (1 << 5);
    return [
        (px & 0xFF) as u8,
        (px >> 8) as u8
    ]
}