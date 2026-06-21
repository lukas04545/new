//! A tiny, dependency-free pseudo-random number generator.
//!
//! This is *not* cryptographically secure — it exists only so the framework
//! can initialize weights and generate toy datasets without pulling in the
//! `rand` crate. It implements `xoshiro256**`, a fast, well-distributed PRNG.

/// A seedable `xoshiro256**` generator.
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Create a generator from a 64-bit seed.
    pub fn new(seed: u64) -> Rng {
        // Expand the seed with SplitMix64 to fill the state.
        let mut sm = seed;
        let mut next = || {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        // Use the top 24 bits for a 24-bit mantissa of precision.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform `f32` in `[lo, hi)`.
    pub fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard-normal `f32` via the Box–Muller transform.
    pub fn normal(&mut self) -> f32 {
        // Avoid log(0) by clamping the first uniform away from zero.
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// A random index in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher–Yates shuffle of a slice.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            slice.swap(i, j);
        }
    }
}
