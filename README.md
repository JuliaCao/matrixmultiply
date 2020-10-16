# DGEMM in Rust

Input matrices are column major

### System
OS: MacOS Catalina 10.15.6

Processor: 2.3 GHz Quad-Core Intel Core i7

Memory: 16 GB 3733 MHz LPDDR4X

## Strategy

### Block Multiply
* Breaks matrices into blocks that fit into l1 cache
* Copies matrices into column major and row major respectively for optimal access pattern
* Performs normal 3 loop once we are inside a small block
* Optimizes for block size 4096
* Achieves ~3x performance improvement
* [Performance report](./benchmark_report/block_vs_naive.pdf)

## Tools
* criterion for benchmark


