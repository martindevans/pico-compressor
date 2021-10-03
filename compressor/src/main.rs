use std::env::{self, set_current_dir};
use std::fs;
use time::Duration;
use std::path::PathBuf;
use structopt::StructOpt;

mod solver;
use solver::Solution;

mod loader;
use loader::Loader;

mod output;
use output::HeaderBuilder;

mod check;
use check::sanity_check;

#[derive(Debug, StructOpt)]
#[structopt(name = "pico-image-compressor", about = "Compress images suitable for scanlie decoding on a Pico.")]
struct Opt
{
    /// Maximum time to spend compressing
    #[structopt(short = "d", long = "duration", default_value = "300000")]
    duration_ms: u32,

    /// Maximum time to spend compressing
    #[structopt(short = "e", long = "early_stopping", default_value = "9")]
    early_stopping: u32,

    /// Input folder
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// Input filter
    #[structopt(short = "f", long = "filter", )]
    filter: String,

    /// Output file, stdout if not present
    #[structopt(parse(from_os_str))]
    output: Option<PathBuf>,
}

fn main()
{
    let opt = Opt::from_args();

    // Find files and load them into memory
    let start_dir = env::current_dir().unwrap();
    set_current_dir(&opt.input).expect("Failed to set cwd to input directory");
    let loader = Loader::create_from_filter(&opt.filter);
    let files = loader.load_files();
    println!("## Loaded images files:");
    for file in files.iter() {
        println!(" - {}", file);
    }
    set_current_dir(&start_dir).expect("Failed to set cwd to original directory");

    let duration = Duration::milliseconds(opt.duration_ms.into());
    let solution = Solution::solve(duration, &files, opt.early_stopping);

    sanity_check(&files, &solution);

    // Setup file output
    let mut header = HeaderBuilder::new();

    header.stats_comment(solution.stats());
    header.image_list_comment(&files);
    header.write_pixels(solution.pixels());
    header.write_metadata(solution.metadata());

    if let Some(output) = opt.output {
        fs::write(output, header.to_string()).unwrap();
    } else {
        println!("{}", header.to_string());
    }
}

