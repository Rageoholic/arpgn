macro_rules! debug_string {
    ($b:expr, $($arg:tt)* ) => {
        $b.then(||format!($($arg)*))
    };
}
pub(crate) use debug_string;
