use std::thread;
use std::time::Instant;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg; // A fast RNG
use std::env;

#[derive(Default)]
struct SimulationResult {
    min_sum: f64,
    max_sum: f64,
}

fn simulate_points(num_simulations: u64) -> SimulationResult {
    let mut rng = Pcg64Mcg::new(random());
    let mut result = SimulationResult::default();

    for _ in 0..num_simulations {
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
            thread::spawn(move || simulate_points(simulations))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();

    let total_result = results.iter().fold(
        SimulationResult::default(),
        |mut acc, res| {
            acc.min_sum += res.min_sum;
            acc.max_sum += res.max_sum;
            acc
        },
    );

    (
        total_result.min_sum / total_simulations as f64,
        total_result.max_sum / total_simulations as f64,
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let total_simulations: u64 = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000_000);

    let num_threads: u64 = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    println!("Running {} simulations with {} thread(s)...", total_simulations, num_threads);

    let start_time = Instant::now();

    let (expected_min, expected_max) = parallel_simulate(total_simulations, num_threads);

    let elapsed_time = start_time.elapsed();

    println!("\n\nSimulation completed in {:.2} seconds", elapsed_time.as_secs_f64());
    println!("Number of simulations: {}", total_simulations);
    println!("Number of threads: {}", num_threads);
    println!("Expected value of minimum point: {:.8}", expected_min);
    println!("Expected value of maximum point: {:.8}", expected_max);
    println!("\nTheoretical expected value of minimum: {:.8}", 1.0 / 3.0);
    println!("Theoretical expected value of maximum: {:.8}", 2.0 / 3.0);
    println!("Difference from theoretical (min): {:.8}", (expected_min - 1.0 / 3.0).abs());
    println!("Difference from theoretical (max): {:.8}", (expected_max - 2.0 / 3.0).abs());
}