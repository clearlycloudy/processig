#[derive(Debug)]
pub struct DistrStat {
    pub mean: f64,
    pub vari: f64,
    pub prec: f64,
}

#[derive(Debug)]
pub struct CachedDistrStat {
    pub prec: Vec< f64 >,
    pub mean: Vec< f64 >,
    pub vari: Vec< f64 >,
    pub prec_weights: Vec< f64 >,
}

impl CachedDistrStat {
    pub fn len( & self ) -> usize {
        self.prec.len()
    }
    pub fn is_valid( & self ) -> bool {
        self.prec.len() == self.mean.len() && self.mean.len() == self.vari.len()
    }
}

impl Default for CachedDistrStat {
    fn default() -> Self {
        Self {
            prec: vec![],
            mean: vec![],
            vari: vec![],
            prec_weights: vec![],
        }
    }
}
