macro_rules! debug_string {
    ($b:expr, $($arg:tt)* ) => {
        $b.then(||format!($($arg)*))
    };
}
pub(crate) use debug_string;

pub fn log2_floor(i: u32) -> u32 {
    let highest_set_bit = u32::BITS - i.leading_zeros();
    highest_set_bit.saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2_floor() {
        for i in 1..2048 {
            assert_eq!((i as f32).log2().floor() as u32, log2_floor(i));
        }
    }
}
