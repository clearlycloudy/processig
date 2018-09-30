use std::ops::Sub;
use std::fmt::Debug;

pub fn nearly_equal<'a, T>( aa: &'a T, bb: & 'a T, ee: & 'a T ) -> bool
    where & 'a T: Sub< Output = T > + PartialOrd,
          T: Sub< Output = T > + Clone + PartialOrd + Debug
{
    let dd = if aa < bb {
        bb-aa
    } else {
        aa-bb
    };
    let e1 = ee.clone();
    dd < e1       
}
