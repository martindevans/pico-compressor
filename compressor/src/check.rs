

use crate::{loader::RawImageData, solver::{Solution, SolutionImage}};

pub fn sanity_check(files: &[RawImageData], solution: &Solution)
{
    let pixels = solution.pixels();
    let metadata = solution.metadata();

    for image in files
    {
        // Find the metadata for this image
        let metadata = metadata.iter().find(|a| a.name == image.name).unwrap();

        let reconstructed = reconstruct_image(&metadata, &pixels, image.width * 2);

        for (idx, row) in image.rows.iter().enumerate()
        {
            if reconstructed[idx] != *row {
                panic!("Reconstruction failure.\nExpected (len:{}): {:?}\nActual (len:{}):{:?}", row.len(), *row, reconstructed[idx].len(), reconstructed[idx]);
            }
        }
    }
}

fn reconstruct_image(metadata: &SolutionImage, pixels: &Vec<u8>, pad_to: usize) -> Vec<Vec<u8>>
{
    let mut rows = Vec::new();

    for row in metadata.rows.iter()
    {
        let idx = row.first_pixel_index as usize;
        let count = row.pixel_count as usize;
        let xpos = row.first_pixel_xpos as usize;

        let mut row = Vec::new();
        for _ in 0..xpos * 2 {
            row.push(0u8);
        }
        row.extend_from_slice(&pixels[idx..][..count]);
        while row.len() < pad_to {
            row.push(0);
        }

        rows.push(row)
    }

    return rows;
}