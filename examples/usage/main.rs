extern crate processig;

use self::processig::kalman;

fn main() {
    const ERROR : f64 = 0.001;
    let distributions = vec![
        &[ 2.3, 4.5, -2.3, 0.4][..],
        &[ 20.3, 8.5, 30.0, 17.5][..],
        &[ -50.3, -51.5, -49.7][..],
    ];
    let mut stats = kalman::calc_distr_stat_from_raw_multi( distributions.as_slice() ).expect("calc stat failed");
    let expected_means = [ 1.2250, 19.0750, -50.5000 ];
    let expected_variances = &[ 8.32917, 78.38917, 0.84 ];
    let expected_precisions = &[ 0.12006, 0.012757, 1.190476 ];

    assert_eq!( stats.len(), 3 );
    stats.mean.iter().zip( expected_means.iter() ).for_each( |(&x0,&x1)| assert!( x0 < x1 + ERROR &&
                                                                                  x0 > x1 - ERROR ) );
    
    stats.vari.iter().zip( expected_variances.iter() ).for_each( |(&x0,&x1)| assert!( x0 < x1 + ERROR &&
                                                                                    x0 > x1 - ERROR ) );
    stats.prec.iter().zip( expected_precisions.iter() ).for_each( |(&x0,&x1)| assert!( x0 < x1 + ERROR &&
                                                                                     x0 > x1 - ERROR ) );
    
    let samples = &[ 1.5, 16.0, -50.5 ];
    let fused = kalman::fuse_weighted_vals( samples, & mut stats ).expect("fusing failed");

    let expected_precision_weights = &[ 0.0907282, 0.0096402, 0.8996315 ];
    assert_eq!( stats.prec_weights.len(), expected_precision_weights.len() );
    stats.prec_weights.iter().zip( expected_precision_weights.iter() ).for_each( |(&x0,&x1)| assert!( x0 < x1 + ERROR &&
                                                                                                      x0 > x1 - ERROR ) );
    
    assert!( fused < -45.141 + ERROR );
    assert!( fused > -45.141 - ERROR );
}
