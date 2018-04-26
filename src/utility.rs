pub fn calc_mean_from_raw( arr: &[f64] ) -> Option< f64 >{
    if arr.len() == 0 {
        None
    } else {
        Some( arr.iter().sum::<f64>() / arr.len() as f64 )
    }
}

pub fn calc_mean_variance_precision_from_raw( arr: &[f64] ) -> Option< ( f64, f64, f64 ) > {
    match calc_mean_from_raw( arr ) {
        Some(m) => {
            if arr.len() <= 1 {
                None
            } else {
                let temp = arr.iter().fold( 0.0, |acc,&x| acc + (x-m)*(x-m) );
                Some( ( m, temp/((arr.len()-1) as f64), ((arr.len()-1) as f64)/temp ) )
            }
        },
        _ => None,
    }
}

pub fn variance_to_precision( num: f64 ) -> f64 {
    1.0/num
}

pub fn calc_precision_weights( precisions: &[f64] ) -> Vec<f64> {
    let sum : f64 = precisions.iter().sum();
    precisions.iter().map( |x| x/sum ).collect()
}


#[test]
fn test_mean(){
    let x = &[ 0.0, 3.5, 99.9, -45.0 ];
    let m = calc_mean_from_raw(x);
    assert!( m.is_some() );
    let n = m.unwrap();
    assert!( n < 14.6 + 0.00000001 );
    assert!( n > 14.6 - 0.00000001 );
}

#[test]
fn test_calc_mean_variance_precision_from_raw(){
    const ERROR : f64 = 0.01;
    let x = &[ 0.0, 3.5, 99.9, -45.0 ];
    let m = calc_mean_variance_precision_from_raw(x);
    assert!( m.is_some() );
    let n = m.unwrap();
    assert!( n.0 < 14.6 + ERROR);
    assert!( n.0 > 14.6 - ERROR);
    assert!( n.1 < 3721.54 + ERROR );
    assert!( n.1 > 3721.54 - ERROR );
    assert!( n.2 < 1./3721.5 + ERROR );
    assert!( n.2 > 1./3721.5 - ERROR );
}
