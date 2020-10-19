use criterion::measurement::{Measurement, ValueFormatter};
use criterion::Throughput;
use std::time::Instant;

// 2.3 GHz * 8 vector width * 2 flops for FMA = 36.8 GF/s
static MAX_SPEED: f64 = 36.8;

pub struct FlopMeasurement {
    n: usize,
    n_iterations: usize,
}

impl FlopMeasurement {
    pub fn new(n: usize, n_iterations: usize) -> Self {
        Self { n, n_iterations }
    }
}

impl Measurement for FlopMeasurement {
    type Intermediate = Instant;
    type Value = f64;

    fn start(&self) -> Self::Intermediate {
        Instant::now()
    }
    fn end(&self, i: Self::Intermediate) -> Self::Value {
        let seconds = i.elapsed();
        let temp = 2e-9 * (self.n_iterations * self.n * self.n * self.n) as f64;
        let Gflops_s = temp / seconds.as_secs_f64();
        Gflops_s
    }
    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        (*v1 + *v2) / 2.
    }

    fn zero(&self) -> Self::Value {
        0f64
    }

    fn to_f64(&self, val: &Self::Value) -> f64 {
        *val
    }
    fn formatter(&self) -> &dyn ValueFormatter {
        &FlopMeasurementFormatter
    }
}

struct FlopMeasurementFormatter;
impl ValueFormatter for FlopMeasurementFormatter {
    fn format_value(&self, value: f64) -> String {
        format!("{}%", value * 100.)
    }

    fn format_throughput(&self, throughput: &Throughput, value: f64) -> String {
        /* Some random impl since we don't care about this method */
        "not implemented".to_string()
    }

    fn scale_values(&self, ns: f64, values: &mut [f64]) -> &'static str {
        "%"
    }

    fn scale_throughputs(
        &self,
        _typical: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        /* Some random impl since we don't care about this method */
        match *throughput {
            Throughput::Bytes(bytes) => {
                // Convert nanoseconds/iteration to bytes/half-second.
                for val in values {
                    *val = (bytes as f64) / (*val * 2f64 * 10f64.powi(-9))
                }

                "b/s/2"
            }
            Throughput::Elements(elems) => {
                for val in values {
                    *val = (elems as f64) / (*val * 2f64 * 10f64.powi(-9))
                }

                "elem/s/2"
            }
        }
    }

    fn scale_for_machines(&self, values: &mut [f64]) -> &'static str {
        "%"
    }
}
