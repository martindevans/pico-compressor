use std::fmt;

use itertools::Itertools;
use kmp::kmp_find;
use memchr::memmem::Finder;
use rand::{Rng, thread_rng};
use time::{Duration, PreciseTime};
use rayon::prelude::*;

use crate::loader::RawImageData;

#[derive(Clone)]
pub struct ImageData {
    pub width: usize,
    pub height: usize,
    pub rows: Vec<ImageRow>,
    pub name: String,
    pub pixel_count: u64,
}

impl fmt::Display for ImageData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:\t{} rows, {} pixels", self.name, self.rows.len(), self.pixel_count)
    }
}

impl ImageData {
    fn new(input: &RawImageData) -> ImageData {
        ImageData {
            width: input.width,
            height: input.height,
            name: input.name.clone(),
            pixel_count: input.pixel_count,
            rows: input.rows.iter().map(|r| ImageRow::new(r)).collect(),
        }
    }
}

#[derive(Clone)]
pub struct ImageRow {
    pub data: Vec<u8>,
    pub alpha_prefix: u16,
    pub finder: Finder<'static>
}

impl fmt::Debug for ImageRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageRow")
         .field("data", &self.data)
         .field("alpha_prefix", &self.alpha_prefix)
         .finish()
    }
}

impl PartialEq for ImageRow {
    fn eq(&self, other: &Self) -> bool
    {    
        if self.alpha_prefix != other.alpha_prefix {
            return false;
        }
        if self.data.len() != other.data.len() {
            return false;
        }

        return self.data.iter().zip_eq(other.data.iter()).all(|(a, b)| *a == *b);
    }
}

impl ImageRow {
    pub fn new(data: &Vec<u8>) -> ImageRow
    {
        let mut data = data.clone();

        // Find how many leading pixels are transparent and remove from row
        let prefix = data.chunks(2).take_while(|px| ImageRow::packed_is_transparent(*px)).count();
        drop(data.drain(0..prefix * 2));

        // Find how many trailing pixels are transparent and remove from row
        let suffix = data.chunks(2).rev().take_while(|px| ImageRow::packed_is_transparent(*px)).count();
        data.truncate(data.len() - suffix * 2);

        let finder = Finder::new(&data).into_owned();

        return ImageRow {
            data,
            alpha_prefix: prefix as u16,
            finder
        };
    }

    fn packed_is_transparent(px: &[u8]) -> bool {
        let px16 = px[0] as u16 | ((px[1] as u16) << 8);
        return px16 & 0b100000 == 0;
    }
}

pub struct SolutionStats {
    pub input_pixel_count: u64,
    pub output_byte_count: u64,
    pub output_metadata_rows: u64,
}

pub struct SolutionRow {
    pub first_pixel_index: u64,
    pub first_pixel_xpos: u64,
    pub is_discontinuous: bool,
    pub pixel_count: u64,
}

pub struct SolutionImage {
    pub rows: Vec<SolutionRow>,
    pub width: usize,
    pub height: usize,
    pub name: String
}

pub struct Solution {
    pixels: Vec<u8>,
    images: Vec<SolutionImage>,
    stats: SolutionStats
}

impl Solution {
    fn new(problem: Problem, solution: Candidate) -> Solution
    {
        let (pixels, offsets) = solution.flatten_with_metadata(1024, &problem.images, problem.trivial);
        let offsets = offsets;

        let input_pixel_count = problem.images.iter().map(|i| i.pixel_count).sum();

        let mut images = Vec::<SolutionImage>::new();
        for (imgidx, image) in problem.images.into_iter().enumerate()
        {
            let rows = offsets.iter()
                .filter(|a| a.imgidx == imgidx)
                .sorted_by_key(|a| a.rowidx)
                .map(|a| {
                    let row = &image.rows[a.rowidx];
                    let is_discontinuous = row.data.chunks(2).any(ImageRow::packed_is_transparent);
                    SolutionRow {
                        first_pixel_index: a.startidx as u64,
                        first_pixel_xpos: row.alpha_prefix as u64,
                        is_discontinuous,
                        pixel_count: row.data.len() as u64
                    }
                })
                .collect();

            images.push(SolutionImage {
                rows,
                width: image.width,
                height: image.height,
                name: image.name,
            });
        }
        
        let stats = SolutionStats {
            input_pixel_count,
            output_byte_count: pixels.len() as u64,
            output_metadata_rows: images.iter().map(|i| i.rows.len()).sum::<usize>() as u64
        };

        return Solution {
            images,
            pixels,
            stats
        }
    }

    pub fn pixels(&self) -> &Vec<u8> {
        &self.pixels
    }

    pub fn metadata(&self) -> &Vec<SolutionImage> {
        &self.images
    }

    pub fn stats(&self) -> &SolutionStats {
        &self.stats
    }
}

#[derive(Clone, Copy)]
struct FlattenedRow {
    imgidx: usize,
    rowidx: usize,
    startidx: usize,
}

#[derive(Clone)]
struct Candidate {
    // Set of (image_index, row_index) tuples
    row_indices: Vec<(usize, usize)>,
}

impl Candidate {
    fn flatten_with_metadata(&self, capacity: usize, images: &Vec<ImageData>, extra_rows: Vec<(usize, usize)>) -> (Vec<u8>, Vec<FlattenedRow>)
    {
        let mut haystack: Vec<u8> = Vec::with_capacity(capacity);
        let mut offsets = Vec::with_capacity(capacity);

        for (imgidx, rowidx) in self.row_indices.iter().chain(extra_rows.iter())
        {
            let row = &images[*imgidx].rows[*rowidx];
            let needle = &row.data;

            let (substring, mut index) = {
                let idx = kmp_find(needle, &haystack);
                (idx.is_some(), idx)
            };

            if !substring
            {
                let overlap = Self::find_overlap(&haystack, needle);
                index = Some(haystack.len() - overlap);
                haystack.extend_from_slice(&needle[overlap..]);
            }

            offsets.push(FlattenedRow { imgidx: *imgidx, rowidx: *rowidx, startidx: index.unwrap() });
        }

        return (haystack, offsets);
    }

    fn flatten(&mut self, capacity: usize, images: &Vec<ImageData>) -> Vec<u8>
    {
        // Keep a "palette" of pairs of value which have been seen so far. Once a byte pair is encountered
        // mark the correspoinding value as true. This can shorcut some substring searches - if the needle
        // contains a byte pair not marked true, it can't be a substring yet.
        let mut palette =  [false; 65536];

        let mut haystack: Vec<u8> = Vec::with_capacity(capacity);

        for (imgidx, rowidx) in self.row_indices.iter()
        {
            let row = &images[*imgidx].rows[*rowidx];

            if !Candidate::is_row_substring(&row, &haystack, &palette)
            {
                let needle = &row.data;
                let overlap = Self::find_overlap(&haystack, needle);

                let length_before = usize::max(10, haystack.len()) - 10;
                haystack.extend_from_slice(&needle[overlap..]);

                for pair in haystack[length_before..].windows(2) {
                    let index = Self::palette_key(pair);
                    palette[index as usize] = true;
                }
            }
        }

        return haystack;
    }

    fn palette_key(pair: &[u8]) -> usize
    {
        return (pair[0] as usize) + ((pair[1] as usize) << 8);
    }

    fn is_row_substring(row: &ImageRow, haystack: &Vec<u8>, palette: &[bool; 65536]) -> bool
    {
        for pair in row.data.windows(2) {
            if !palette[Self::palette_key(pair)] {
                return false;
            }
        }

        return row.finder.find_iter(haystack).next().is_some();
    }

    // Find how much of the end of the left overlaps with the start of the right
    fn find_overlap(left: &[u8], right: &[u8]) -> usize 
    {
        for i in (0..right.len()).rev()
        {
            if left.ends_with(&right[0..i])
            {
                return i;
            }
        }
        return 0;
    }
}

struct Problem {
    images: Vec<ImageData>,
    max_size: usize,
    prototype: Vec<(usize, usize)>,
    trivial: Vec<(usize, usize)>
}

impl Problem
{
    fn new(input: &Vec<RawImageData>) -> Problem
    {
        let images = input.iter()
            .map(|i| ImageData::new(i))
            .collect::<Vec<_>>();

        let max_size = images.iter().map(|i| i.pixel_count * 2).sum::<u64>() as usize;
        
        let all_rows = images.iter()
            .flat_map(|i| &i.rows)
            .map(|m| m.data.clone())
            .collect::<Vec<_>>();

        let mut index = 0;
        let mut prototype = Vec::new();
        let mut trivial = Vec::new();
        for (imgidx, img) in images.iter().enumerate()
        {
            for (rowidx, row) in img.rows.iter().enumerate()
            {
                if !Problem::is_trivial(&row.data, index, &all_rows)
                {
                    prototype.push((imgidx, rowidx));
                }
                else
                {
                    trivial.push((imgidx, rowidx));
                }
                index += 1;
            }
        }

        println!("Found {} trivial rows, {} remain", trivial.len(), prototype.len());

        Problem {
            images,
            max_size,
            prototype,
            trivial,
        }
    }

    // Check if the given row is trivially a subset because it's a subset of another
    // row! We can exclude this from the search set because it's guaranteed to be
    // included already.
    fn is_trivial(row: &Vec<u8>, row_index: usize, rows: &Vec<Vec<u8>>) -> bool
    {
        return rows
            .par_iter()
            .enumerate()
            .any(|(idx, r)| idx != row_index && kmp_find(&row, &r).is_some());
    }

    fn generate_candidate(&self) -> Candidate
    {
        let mut indices = self.prototype.clone();
        thread_rng().shuffle(&mut indices);

        Candidate {
            row_indices: indices,
        }
    }

    fn tweak_candidate(&self, mut candidate: Candidate, temperature: f32) -> Candidate
    {
        if candidate.row_indices.len() <= 1 {
            return candidate;
        }
        
        let mut rng = thread_rng();
        for _ in 0..(temperature as u32)
        {
            let len = candidate.row_indices.len();
            let idx0 = rng.gen_range::<usize>(0, len);
            let idx1 = rng.gen_range::<usize>(0, len);
            candidate.row_indices.swap(idx0, idx1);
        }

        return candidate;
    }

    fn rank_candidate(&self, candidate: &mut Candidate) -> usize
    {
        let data = candidate.flatten(self.max_size, &self.images);
        return data.len();
    }
}

impl Solution {
    pub fn solve(runtime: Duration, images: &Vec<RawImageData>) -> Solution {
        let mut problem = Problem::new(images);
        let solution = custom_solve(&mut problem, 64, 999, runtime);
        return Solution::new(problem, solution);
    }
}

fn custom_solve(problem: &mut Problem, items: usize, max_items: usize, runtime: Duration) -> Candidate
{
    // Generate the initial guess
    let initial_best = (0..items+1).into_par_iter()
        .map(|_| problem.generate_candidate())
        .map(|mut c| (problem.rank_candidate(&mut c), c))
        .min_by_key(|x| x.0)
        .expect("Expected at least 1 item in round");

    let mut best_candidate = initial_best.1;
    let mut best_score = initial_best.0;

    let mut pool = Vec::new();
    for _ in 0..items-1 {
        pool.push(best_candidate.clone());
    }

    let mut temperature = 293f32;
    let mut total_samples: u64 = 0;
    let start_time = PreciseTime::now();
    while start_time.to(PreciseTime::now()) < runtime
    {
        // Tweak every item in pool
        total_samples += pool.len() as u64;
        let mut results = pool.into_par_iter()
            .map(|c| problem.tweak_candidate(c, temperature))
            .map(|mut c| (problem.rank_candidate(&mut c), c))
            .collect::<Vec<_>>();
            results.sort_unstable_by_key(|x| x.0);

        let best_in_round = &results[0];
        if best_in_round.0 < best_score
        {
            let best_in_round = results.into_iter().nth(0).expect("No item in round!");
            best_score = best_in_round.0;
            best_candidate = best_in_round.1;

            // Fill pool with copies of best candidate and one wildcard
            pool = (0.. items - 1)
                .map(|_| best_candidate.clone())
                .take(items - 1)
                .chain([ problem.generate_candidate() ])
                .collect();
        }
        else
        {
            // This wasn't an improvement, fill pool with all the results from the
            // last round plus the same number of copies of the best so far.
            pool = results.into_iter()
                .sorted_unstable_by_key(|k| -(k.0 as i64))
                .flat_map(|item| { [ item.1, best_candidate.clone() ] })
                .take(max_items)
                .collect();

            // Lower the temperature slightly
            temperature = f32::max(temperature * 0.85, 1f32)
        }

        let ratio = (best_score as f64 / problem.max_size as f64) * 100f64;
        println!("score:{} ({:.3}%) pool_size:{} temp:{:.1}K", best_score, ratio, pool.len(), temperature);
    }

    println!("Total tested: {}", total_samples);
    return best_candidate;
}

#[cfg(test)]
mod tests {
     use super::*;

    #[test]
    fn no_overlap() {
        assert_eq!(0, Candidate::find_overlap(&[1,2,3,4,5,6], &[7,8,9]));
    }
 
    #[test]
    fn small_overlap() {
        assert_eq!(1, Candidate::find_overlap(&[1,2,3,4,5,6], &[6,7,8,9]));
    }

    #[test]
    fn large_overlap() {
        assert_eq!(3, Candidate::find_overlap(&[1,2,3,4,5,6], &[4,5,6,7]));
    }

    #[test]
    fn substr() {

        let haystack = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        let needle = vec![1u8, 2, 3, 4];

        let searcher = unsafe { DynamicAvx2Searcher::new(needle) };
        assert!(unsafe {
            searcher.search_in(&haystack)
        });
    }

    #[test]
    fn no_substr() {

        let haystack = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        let needle = vec![1u8, 2, 3, 4, 6];

        let searcher = unsafe { DynamicAvx2Searcher::new(needle) };
        assert!(unsafe {
            !searcher.search_in(&haystack)
        });
    }
}