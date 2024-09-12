use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::arch::x86_64::*;
use std::env;
use std::thread;
use std::time::Instant;

#[derive(Default)]
struct SimulationResult {
    min_sum: f64,
    max_sum: f64,
}

#[target_feature(enable = "avx2")]
unsafe fn simulate_points_avx2(num_simulations: u64, seed: u64) -> SimulationResult {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut result = SimulationResult::default();

    let iterations = num_simulations / 4;
    let remainder = num_simulations % 4;

    let mut min_sum = _mm256_setzero_pd();
    let mut max_sum = _mm256_setzero_pd();

    for _ in 0..iterations {
        let r1: f64 = rng.gen();
        let r2: f64 = rng.gen();
        let r3: f64 = rng.gen();
        let r4: f64 = rng.gen();
        let r5: f64 = rng.gen();
        let r6: f64 = rng.gen();
        let r7: f64 = rng.gen();
        let r8: f64 = rng.gen();

        let vec1 = _mm256_set_pd(r1, r2, r3, r4);
        let vec2 = _mm256_set_pd(r5, r6, r7, r8);

        let min_vec = _mm256_min_pd(vec1, vec2);
        let max_vec = _mm256_max_pd(vec1, vec2);

        min_sum = _mm256_add_pd(min_sum, min_vec);
        max_sum = _mm256_add_pd(max_sum, max_vec);
    }

    let mut min_array = [0.0; 4];
    let mut max_array = [0.0; 4];
    _mm256_storeu_pd(min_array.as_mut_ptr(), min_sum);
    _mm256_storeu_pd(max_array.as_mut_ptr(), max_sum);

    result.min_sum = min_array.iter().sum();
    result.max_sum = max_array.iter().sum();

    // Handle remaining simulations
    for _ in 0..remainder {
        let point1: f64 = rng.gen();
        let point2: f64 = rng.gen();
        result.min_sum += point1.min(point2);
        result.max_sum += point1.max(point2);
    }

    result
}

fn parallel_simulate(total_simulations: u64, num_threads: u64) -> (f64, f64) {
    let chunk_size = total_simulations / num_threads;
    let remainder = total_simulations % num_threads;

    let results: Vec<_> = (0..num_threads)
        .map(|i| {
            let simulations = if i == num_threads - 1 {
                chunk_size + remainder
            } else {
                chunk_size
            };
            let seed = thread_rng().next_u64();
            thread::spawn(move || unsafe { simulate_points_avx2(simulations, seed) })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();

    let total_result = results
        .iter()
        .fold(SimulationResult::default(), |mut acc, res| {
            acc.min_sum += res.min_sum;
            acc.max_sum += res.max_sum;
            acc
        });

    (
        total_result.min_sum / total_simulations as f64,
        total_result.max_sum / total_simulations as f64,
    )
}

fn parse_args() -> (u64, u64) {
    let args: Vec<String> = env::args().collect();
    let mut total_simulations = 100_000_000;
    let mut num_threads = 1;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-s" | "--simulations" => {
                if i + 1 < args.len() {
                    total_simulations = args[i + 1].parse().unwrap_or(100_000_000);
                    i += 1;
                }
            }
            "-t" | "--threads" => {
                if i + 1 < args.len() {
                    num_threads = args[i + 1].parse().unwrap_or(1);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    (total_simulations, num_threads)
}

fn main() {
    let (total_simulations, num_threads) = parse_args();

    println!(
        "Running {} simulations with {} thread(s)...",
        total_simulations, num_threads
    );

    let start_time = Instant::now();

    let (expected_min, expected_max) = parallel_simulate(total_simulations, num_threads);

    let elapsed_time = start_time.elapsed();

    println!(
        "\nSimulation completed in {:.2} seconds",
        elapsed_time.as_secs_f64()
    );
    println!("Number of simulations: {}", total_simulations);
    println!("Number of threads: {}", num_threads);
    println!("Expected value of minimum point: {:.8}", expected_min);
    println!("Expected value of maximum point: {:.8}", expected_max);
    println!("\nTheoretical expected value of minimum: {:.8}", 1.0 / 3.0);
    println!("Theoretical expected value of maximum: {:.8}", 2.0 / 3.0);
    println!(
        "Difference from theoretical (min): {:.8}",
        (expected_min - 1.0 / 3.0).abs()
    );
    println!(
        "Difference from theoretical (max): {:.8}",
        (expected_max - 2.0 / 3.0).abs()
    );
}
