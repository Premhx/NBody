use rand::Rng;
use std::fs::File;
use std::io::{self, Write};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

const N: usize = 50000;
const STEPS: usize = 3000;
const DT: f64 = 0.0001;

#[derive(Clone, Copy)]
struct Body {
    pos: [f64; 2],
    vel: [f64; 2],
    mass: f64,
}

fn rand_double() -> f64 {
    rand::thread_rng().gen::<f64>()
}

fn rand_disc(pairs: &mut [f64; 2]) {
    let theta = rand_double() * 2.0 * PI;
    pairs[0] = theta.cos() * (rand_double()).sqrt();
    pairs[1] = theta.sin() * (rand_double()).sqrt();
}

fn rand_body() -> Body {
    let mut body = Body {
        pos: [0.0; 2],
        vel: [0.0; 2],
        mass: 0.0,
    };
    rand_disc(&mut body.pos);
    rand_disc(&mut body.vel);
    body.mass = rand_double();
    body
}

fn update_bodies(bodies: &mut [Body], dt: f64) {
    let d_min = 0.0001;

    let acc: Vec<[f64; 2]> = vec![[0.0; 2]; bodies.len()];

    let acc = Arc::new(Mutex::new(acc));

    bodies.par_iter().enumerate().for_each(|(i, body1)| {
        let mut acc = acc.lock().unwrap();
        for (j, body2) in bodies.iter_mut().enumerate().skip(i + 1) {
            let r = [body2.pos[0] - body1.pos[0], body2.pos[1] - body1.pos[1]];
            let mag_sq = r[0].powi(2) + r[1].powi(2);
            let mag = f64::max(mag_sq.sqrt(), d_min);
            let tmp = [r[0] / (mag_sq * mag), r[1] / (mag_sq * mag)];
            acc[i][0] += body2.mass * tmp[0];
            acc[i][1] += body2.mass * tmp[1];
            acc[j][0] -= body1.mass * tmp[0];
            acc[j][1] -= body1.mass * tmp[1];
        }
    });

    let mut acc = acc.lock().unwrap();
    bodies.iter_mut().zip(acc.iter_mut()).for_each(|(body, a)| {
        body.vel[0] += a[0] * dt;
        body.vel[1] += a[1] * dt;
        body.pos[0] += body.vel[0] * dt;
        body.pos[1] += body.vel[1] * dt;
    });
}

fn simulate_and_save(seed: u64, steps: usize, filename: &str) -> io::Result<()> {
    let mut rng = rand::thread_rng();
    let mut bodies = Vec::with_capacity(N);
    for _ in 0..N {
        bodies.push(rand_body());
    }

    let mut file = File::create(filename)?;

    let progress_interval = steps / 100;

    for i in 0..steps {
        for body in &bodies {
            writeln!(file, "{},{}", body.pos[0], body.pos[1])?;
        }
        writeln!(file)?;
        update_bodies(&mut bodies, DT);

        if (i + 1) % progress_interval == 0 || i == steps - 1 {
            let progress = (i + 1) * 100 / steps;
            println!("Progress: {}%  ETA: {:.2} seconds", progress, (100 - progress) as f64 / progress as f64 * DT * i as f64);
        }
    }

    Ok(())
}

fn main() -> io::Result<()> {
    simulate_and_save(3, STEPS, "simulation_coords.csv")
}
