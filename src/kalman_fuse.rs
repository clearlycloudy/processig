/// vector data fusion proportional to  precision weights

//todo: try use ndarray
use ndarray::prelude::*;

use utility;
use error::*;
use mazth::mat::*;

use std::ops::{Index, IndexMut};
use std::cmp::min;

#[derive(Debug, Default)]
pub struct Stats {
    pub avgs: Vec< Vec< f64 > >,
    pub mat_cov: Vec< Mat4< f64 > >,
    pub mat_prec: Vec< Mat4< f64 > >,
}

#[derive(Debug, Default)]
pub struct Weights {
    weights: Vec< Mat4< f64 > >,
}

///calculate mean, covariance, etc
pub fn vector_calc_stats( input_sample: &Vec< Vec< Vec< f64 > > > ) -> Result< Stats, Error > {

    // data validity check starts

    let num_sources = input_sample.len();
    if num_sources == 0 { return Err( Error::DataInsufficient ) }
    
    let sample_length = match input_sample.iter().nth(0) {
        Some(x) => x.len(),
        _ => { return Err( Error::DataInsufficient ) },
    };

    let data_vector_length = match input_sample.iter().nth(0).unwrap().iter().nth(0) {
        Some(x) => x.len(),
        _ => { return Err( Error::DataInsufficient ) },            
    };

    for i in input_sample.iter() {
        if i.len() != sample_length { return Err( Error::DataInsufficient ) }
        for j in i.iter() {
            if data_vector_length != j.len() {
                return Err( Error::DataInsufficient )
            }
        }
    }

    // data validity check ends


    //temporarily enforce vector data length to be 4 or greater
    if data_vector_length < 4 { return Err( Error::DataInsufficient ) }
    
    let mut s = Stats::default();

    // calculate mean

    let averages = input_sample.iter()
        .map( |x| {

            let mut avgs = x.iter().fold( vec![0.; 4], |mut acc, ref samples| {
                for j in 0..4 {
                    acc[j] += samples[j];
                }
                acc
            });
            
            for i in avgs.iter_mut() {
                *i = *i / sample_length as f64;
            }
            
            avgs

        }).collect::< Vec< Vec< f64 > > >();

    s.avgs = averages;
    
    s.mat_cov.resize( num_sources, Mat4::default() );
    s.mat_prec.resize( num_sources, Mat4::default() );
    
    // calculate covariance and precision matrices

    for i in 0..num_sources {

        for j in 0..data_vector_length {
            for k in j..data_vector_length {

                let mut a = 0.;
                
                for data_source in input_sample[i].iter() {

                    let x = data_source[j];
                    let y = data_source[k];
                    let temp = (x-s.avgs[i][j])*(y-s.avgs[i][k]);
                    a = a + temp;
                }

                let covariance = a / ( (sample_length-1) as f64 );
                let precision = ( (sample_length-1) as f64 ) / a;

                *s.mat_cov[i].index_mut( j as _, k as _) = covariance;
                *s.mat_cov[i].index_mut( k as _, j as _) = covariance;
                
                *s.mat_prec[i].index_mut( j as _, k as _) = precision;
                *s.mat_prec[i].index_mut( k as _, j as _) = precision;
            }
        }

        match s.mat_cov[i].inverse() {
            Some(inv) => {
                s.mat_prec[i] = inv;
            },
            _ => {
                println!("inv_covariance for source {} is degenerate", i );
                return Err( Error::DataInvalid )
            }
        }
    }

    Ok( s )
}

pub fn vector_calc_weights( s: &Stats ) -> Result< Weights, Error > {
                                         
    // calculate the weights from precision matrices

    let mut op_invalid = false;
    let sum_precisions = s.mat_prec.iter().fold( Mat4::< f64 >::default(), |acc, x| {
        match acc.plus( x ) {
            Some(ret) => ret,
            _ => {
                op_invalid = true;
                acc
            },
        }
    });

    if op_invalid {
        return Err( Error::Operation )
    }

    let weight_normalization = match sum_precisions.inverse() {
        Some(inv) => inv,
        _ => { return Err( Error::DataInvalid ) },
    };


    // assume data source is pairwise uncorrelated and we use the following to weigh data sources
    // weight_i = ( sum_all( precisions ) )^-1 * precision_i, where i is for each data source

    let weights = s.mat_prec.iter().map( |&x| {
        weight_normalization.mul( &x ).unwrap()
    }).collect::< Vec< Mat4< f64 > > >();

    Ok( Weights { weights: weights } )
}

pub fn vector_fuse_data( w: &Weights, input_data: &Vec< Vec< f64 > > ) -> Result< Vec<f64>, Error > {

    let num_sources = w.weights.len();

    if num_sources == 0 || num_sources != input_data.len() { return Err( Error::DataInsufficient ) }

    let data_vector_length = input_data[0].len();

    for i in input_data.iter() {
        if i.len() != data_vector_length {
            return Err( Error::Dimension )
        }
    }

    let data = input_data.iter().map(|x| {

        let mut m = Mat4x1::default();

        //current implementation truncates to use first 4 elements of the data vector
        for i in 0.. min(4, data_vector_length) {
            m._val[i] = x[i];
        }
        
        m
            
    }).collect::< Vec< Mat4x1< f64 > > >();

    let fused = w.weights.iter().zip( data.iter() ).fold( Mat4x1::<f64>::default(), |acc, (&weight, &data)| {
        acc.plus( &weight.mul_mat4x1( &data ).unwrap() ).unwrap()
    });

    let mut ret = vec![ 0.; data_vector_length ];

    //current implementation truncates to use first 4 elements of the data vector
    for i in 0..4 {
        ret[i] = fused._val[i];
    }

    Ok( ret )
}

#[test]
fn test_fuse() {
    
    /// test with 3 data sources each with vector length of 4
    
    /// generate sample data with the following distributions
    let mu0 = 2.0;
    let sigma0 = 1.0;
    let sample0 = [
        [  2.8884,  1.8978,  1.1363,  0.9109, ],
        [  0.8529,  1.7586,  2.0774,  2.0326, ],
        [  0.9311,  2.3192,  0.7859,  2.5525, ],
        [  1.1905,  2.3129,  0.8865,  3.1006, ],
        [ -0.9443,  1.1351,  1.9932,  3.5442, ],
        [  3.4384,  1.9699,  3.5326,  2.0859, ],
        [  2.3252,  1.8351,  1.2303,  0.5084, ],
        [  1.2451,  2.6277,  2.3714,  1.2577, ],
        [  3.3703,  3.0933,  1.7744,  0.9384, ],
        [  0.2885,  3.1093,  3.1174,  4.3505, ],
    ].into_iter().map(|x| x.to_vec() ).collect::< Vec< Vec< f64 > > >();
    
    let mu1 = 10.0;
    let sigma1 = 5.0;
    let sample1 = [
        [  17.0966,  4.2602,   14.2019,  -0.6918, ],
        [  11.4579,  10.5244,  5.5598,   5.8021,  ],
        [  10.9891,  13.6113,  10.5005,  16.7730, ],
        [  17.9385,  22.9275,  7.2774,   4.6392,  ],
        [  5.9777,   6.6655,   11.5176,  14.8048, ],
        [  13.4831,  10.9367,  6.9984,   10.6202, ],
        [  14.1754,  9.5875,   12.4498,  17.1835, ],
        [  8.7814,   0.3349,   13.6968,  0.1955,  ],
        [  11.0784,  7.8052,   18.5594,  9.0115,  ],
        [  4.1708,   1.0266,   9.0294,   3.9608,  ],
    ].into_iter().map(|x| x.to_vec() ).collect::< Vec< Vec< f64 > > >();

    let mu2 = -3.0;
    let sigma2 = 2.0;
    let sample2 = [
        [ -3.7077,   -2.9542,   -1.9599,   -3.5875, ],
        [ -4.6472,   -3.5240,   -3.0401,   -4.6959, ],
        [ -6.1541,   -6.5004,   -3.0695,   -5.2403, ],
        [ -1.9841,   -3.5713,   -4.5963,    2.0520, ],
        [ -2.4360,   -4.6627,   -0.9626,    0.3110, ],
        [ -2.9330,   -4.9584,   -3.2664,   -2.3849, ],
        [ -5.6674,   -5.3128,   -4.4291,   -5.5142, ],
        [ -0.7450,   -4.0671,   -0.2972,   -4.7309, ],
        [ -2.2996,   -7.0053,   -3.4495,   -3.3531, ],
        [  -3.5981,  -1.0715,   -4.1781,   -1.4172, ],
    ].into_iter().map(|x| x.to_vec() ).collect::< Vec< Vec< f64 > > >();
    
    let sample_input_vectors = vec![
        sample0,
        sample1,
        sample2,
    ];
    
    let stats = match vector_calc_stats( & sample_input_vectors ) {
        Ok( s ) => {
            println!( "{:#?}", s );
            s
        },
        Err( e ) => { panic!(e); }
    };
    
    let weights = match vector_calc_weights( & stats ) {
        Ok( w ) => w,
        Err( e ) => { panic!(e); },
    };

    println!( "weights: {:#?}", weights );

    let data = [
        [  0.9,   2.3,   0.7,   2.5, ], //data source 1
        [  10.9,  13.6,  10.5,  16.7, ], //data source 2
        [ -3.5,   -5.,    -2.5,  -6., ], ////data source 3
    ].into_iter().map(|x| x.to_vec() ).collect::< Vec< Vec< f64 > > >();
    
    let fused_data = match vector_fuse_data( & weights, & data ) {
        Ok( d ) => d,
        Err( e ) => { panic!(e); },
    };

    println!( "fused_data: {:#?}", fused_data );

    const ERROR : f64 = 0.001;
    
    assert!( fused_data[0] > 1.2825 - ERROR );
    assert!( fused_data[0] < 1.2825 + ERROR );

    assert!( fused_data[1] > 1.6683 - ERROR );
    assert!( fused_data[1] < 1.6683 + ERROR );

    assert!( fused_data[2] > 0.0237 - ERROR );
    assert!( fused_data[2] < 0.0237 + ERROR );

    assert!( fused_data[3] > 1.4724 - ERROR );
    assert!( fused_data[3] < 1.4724 + ERROR );
}

