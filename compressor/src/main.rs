use std::env::set_current_dir;
use std::fs;
use std::path::Path;
use time::Duration;

use clap::{ Arg, App };

mod solver;
use solver::Solution;

mod loader;
use loader::Loader;

mod output;
use output::HeaderBuilder;

fn main() {
    let matches = App::new("Pico Image Compressor")
        .version("1.0")
        .author("Martin Evans")
        .about("Compresses a set of images in a way that can be decompressed on the Pico")
        .arg(Arg::with_name("INPUT")
            .help("Sets the input folder to use")
            .required(true)
            .index(1))
        .arg(Arg::with_name("OUTPUT")
            .help("Sets the output file to use")
            .required(true)
            .index(2))
        .arg(Arg::with_name("FILTER")
            .long("filter")
            .help("Sets the file filter to use")
            .required(false)
            .default_value("**/*.png"))
        .arg(Arg::with_name("DURATION")
            .long("duration")
            .help("Sets the duration to compress for (in ms)")
            .required(false)
            .default_value("30000"))
        .get_matches();

    set_current_dir(Path::new(matches.value_of("INPUT").expect("INPUT had no value.")))
        .expect("Failed to set cwd to input directory");

    // Setup file output
    let mut header = HeaderBuilder::new();

    // Find files and load them into memory
    let loader = Loader::create_from_filter(matches.value_of("FILTER").expect("FILTER had no value."));
    let files = loader.load_files();
    println!("## Loaded images files:");
    for file in files.iter() {
        println!(" - {}", file);
    }

    let milliseconds: u32 = matches.value_of("DURATION").expect("DURATION had no value").parse::<u32>().unwrap_or(30000);
    let duration = Duration::milliseconds(milliseconds as i64);
    let solution = Solution::solve(duration, files.clone());

    header.stats_comment(solution.stats());
    header.image_list_comment(&files);
    header.write_pixels(solution.pixels());
    header.write_metadata(solution.metadata());

    let output = matches.value_of("OUTPUT").expect("OUTPUT had no value");
    fs::write(output, header.to_string()).unwrap();
}