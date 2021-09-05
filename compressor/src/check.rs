use crate::solver::{ImageData, ImageRow, Solution, SolutionImage};

pub fn sanity_check(files: &[ImageData], solution: &Solution)
{
    let pixels = solution.pixels();
    let metadata = solution.metadata();

    for image in files
    {
        // Find the metadata for this image
        let metadata = metadata.iter().find(|a| a.name == image.name).unwrap();

        let reconstructed = reconstruct_image(&metadata, &pixels);

        for (idx, row) in image.rows.iter().enumerate()
        {
            if reconstructed[idx] != *row {
                panic!("Reconstruction failure.\nExpected: {:?}\nActual:{:?}", *row, reconstructed[idx]);
            }
        }
    }
}

fn reconstruct_image(metadata: &SolutionImage, pixels: &Vec<u8>) -> Vec<ImageRow>
{
    let mut rows = Vec::new();

    for row in metadata.rows.iter()
    {
        let idx = row.first_pixel_index as usize;
        let count = row.pixel_count as usize;
        let xpos = row.first_pixel_xpos as usize;

        let data = pixels[idx..][..count].to_vec();
        let mut row = ImageRow::new(data);
        row.alpha_prefix = xpos as u16;

        rows.push(row)
    }

    return rows;
}