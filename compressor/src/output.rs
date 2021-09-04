use std::time::{SystemTime, UNIX_EPOCH};

use itertools::Itertools;

use crate::solver::{ImageData, SolutionImage, SolutionStats};

pub struct HeaderBuilder {
    strings: Vec<String>,
    uid: String
}

impl HeaderBuilder {
    pub fn new() -> HeaderBuilder
    {
        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let ms = elapsed.as_secs() * 1000 + elapsed.subsec_nanos() as u64 / 1_000_000;

        let mut out = HeaderBuilder {
            strings: Vec::new(),
            uid: ms.to_string()
        };

        out.push("#include <stdint.h>");
        out.push("#include \"sprite.h\"");
        out.push("");

        return out;
    }

    pub fn push<T: ToString>(&mut self, str: T) {
        self.strings.push(str.to_string());
    }

    pub fn image_list_comment(&mut self, images: &Vec<ImageData>)
    {
        self.push("// ## Loaded images:");
        for image in images.iter().sorted_by_key(|a| a.pixel_count).rev() {
            self.push(format!("// - {0} ({1} pixels)", &image.name, image.pixel_count));
        }
        self.push("");
    }

    pub fn stats_comment(&mut self, stats: &SolutionStats)
    {
        let input_pixels = stats.input_pixel_count;
        let output_bytes = stats.output_byte_count;
        let output_metadata_bytes = stats.output_metadata_rows * 5;

        self.push("// ## Compression Stats:");

        let saved_bytes = input_pixels * 2 - output_bytes;
        self.push(format!("// - {0} bytes of pixel data ({1} bytes saved)", output_bytes, saved_bytes));
        self.push(format!("// - {0} bytes of metadata added", output_metadata_bytes));
        self.push(format!("// - Reduced {0} bytes to {1} bytes", input_pixels * 2, output_bytes));

        let ratio = ((input_pixels * 2) as f32) / ((output_bytes + output_metadata_bytes) as f32);
        self.push(format!("// - {0} compression ratio", ratio));

        let amount = 1f32 / ratio;
        self.push(format!("// - {0}x original size", amount));
        self.push("");
    }

    pub fn write_pixels(&mut self, pixels: &Vec<u8>)
    {
        self.push(format!("static uint8_t pixels_{0}[] = {{", self.uid));

        for chunk in &pixels.iter().chunks(35) {
            self.push("    ".to_owned() + &chunk.map(|px| format!("{0:0>3}", px)).join(","));
        }
        self.push("};");
    }

    pub fn write_metadata(&mut self, metadata: &Vec<SolutionImage>)
    {
        for image in metadata
        {
            self.push(format!("static const row_metadata_t {}_metadata[] = {{", image.name));
            for row in image.rows.iter()
            {
                self.push("    {");
                self.push(format!("        .index_first_byte = {},", row.first_pixel_index));
                self.push(format!("        .first_pixel_xpos = {},", row.first_pixel_xpos));
                self.push(format!("        .is_discontinuous = {},", row.is_discontinuous));
                self.push(format!("        .pixel_byte_count = {},", row.pixel_count));
                self.push("    },");
            }
            self.push("};");
        }

        for image in metadata
        {                
            self.push(format!("static const image_data_t {0}_{1}x{2} = {{", image.name, image.width, image.height));
            self.push(format!("    .pixels = pixels_{0},", self.uid));
            self.push(format!("    .metadata = {0}_metadata,", image.name));
            self.push(format!("    .size_x = {0},", image.width));
            self.push(format!("    .size_y = {0},", image.height));
            self.push("};");
        }
    }
}

impl ToString for HeaderBuilder {
    fn to_string(&self) -> String {
        return self.strings.join("\n");
    }
}