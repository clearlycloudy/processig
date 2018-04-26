use utility;
use error;
use distribution::*;

pub fn fuse_weighted_vals( samples: &[f64], distr_stats: &mut CachedDistrStat )-> Result< f64, error::Error > {

    if samples.len() != distr_stats.len() {
        Err( error::Error::Dimension )
    } else {
        let precision_weights = utility::calc_precision_weights( distr_stats.prec.as_slice() );
        distr_stats.prec_weights = precision_weights.clone();
        Ok( samples.iter().zip( precision_weights.iter() ).fold( 0.0, |accum, (&val, &weight) | accum + val * weight ) )
    }
}

pub fn calc_distr_stat_from_raw_multi( arr_arr: &[ &[f64] ] ) -> Result< CachedDistrStat, error::Error > {
    let mut cached_stat = CachedDistrStat::default();
    for &i in arr_arr.iter() {
        let stat = calc_distr_stat_from_raw(i)?;
        add_distr_stat_to_cache( & mut cached_stat, &stat );
    }
    Ok( cached_stat )
}

pub fn calc_distr_stat_from_raw( arr: &[f64] ) -> Result< DistrStat, error::Error > {
    match utility::calc_mean_variance_precision_from_raw( arr ) {
        Some(x) => {
            Ok( DistrStat {
                mean: x.0,
                vari: x.1,
                prec: x.2,
            } )
        },
        _ => {
            Err( error::Error::DataEmpty )
        },
    }
}

pub fn add_distr_stat_to_cache( cache: & mut CachedDistrStat, distr_stat: & DistrStat ){
    cache.prec.push(distr_stat.prec);
    cache.mean.push(distr_stat.mean);
    cache.vari.push(distr_stat.vari);
}


#[test]
fn test_fuse() {
    const ERROR : f64 = 0.001;
    let distributions = vec![
        &[ 2.3, 4.5, -2.3, 0.4][..],
        &[ 20.3, 8.5, 30.0, 17.5][..],
        &[ -50.3, -51.5, -49.7][..],
    ];
    let mut stats = calc_distr_stat_from_raw_multi( distributions.as_slice() ).expect("calc stat failed");
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
    let fused = fuse_weighted_vals( samples, & mut stats ).expect("fusing failed");

    let expected_precision_weights = &[ 0.0907282, 0.0096402, 0.8996315 ];
    assert_eq!( stats.prec_weights.len(), expected_precision_weights.len() );
    stats.prec_weights.iter().zip( expected_precision_weights.iter() ).for_each( |(&x0,&x1)| assert!( x0 < x1 + ERROR &&
                                                                                                      x0 > x1 - ERROR ) );
    
    assert!( fused < -45.141 + ERROR );
    assert!( fused > -45.141 - ERROR );
}
