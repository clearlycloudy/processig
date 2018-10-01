///Decompose a signal into a combination of dictionary supports that maybe not be orthogonal.
///This uses greedy approach using dot product metric that tries to reduce error vector per iteration
extern crate ndarray;

use std::vec::Vec;

use self::ndarray::prelude::*;

///stopping criteria
pub struct Criteria {
    pub error_threshold: Option<f64>,
    pub num_max_support: Option<usize>,
}

pub enum Error {
    ErrRemain(f64),
    SupportInvalid,
    Other,
}

///result is a vector of ( indicies, scaling coefficient ) corresponding to the input dictionary
pub struct Sparse( pub Vec< ( usize, f64 ) > );

#[derive(Clone)]
pub enum Signal {
    Discrete( ndarray::Array1<f64> ),
}

fn len( s: & Signal ) -> usize {
    match s {
        Signal::Discrete(x) => {
            x.len()            
        },
        _ => { unimplemented!(); },
    }
}

fn magnitude( s: & Signal ) -> f64 {
    dot( &s, &s )
}

fn interp_discrete( a: & Signal, len: usize ) -> Signal {
    unimplemented!();
}

fn dot( a: & Signal, b: & Signal ) -> f64 {

    use std::cmp;
    let len_max = cmp::max( len(a), len(b) );

    let mut temp_a : Signal;
    let mut temp_b : Signal;
    let ( x_adjust, y_adjust ) = if len(a) != len(b) {
        temp_a = interp_discrete( a, len_max );
        temp_b = interp_discrete( b, len_max );
        ( &temp_a, &temp_b )
    } else {
        ( a, b )
    };

    match ( x_adjust, y_adjust ) {
        ( Signal::Discrete(x), Signal::Discrete(y) ) => {
            let mut accum = 0.0;
            for (v0,v1) in x.iter().zip( y.iter() ) {
                accum += v0 * v1;
            }
            accum
        },
        _ => {
            unimplemented!();
        },
    }
}

///Vanilla matching pursuit implementation.
///Computes the sparse combination using dictionary supports and returns the sparse result and residual error on success.
///Errors out if error threshold is not met
pub fn matching_pursuit( dict: & Vec< Signal >, s: &Signal, criteria: & Criteria ) -> Result< ( Sparse, f64 ), Error > {
    let mut used = vec![ None; dict.len() ];
    let mut used_count = 0;
    let mut error = s.clone();
    let mut sparse = Sparse(vec![]);
    loop {
        match criteria.error_threshold {
            Some(x) if magnitude(&error) < x => {
                break;
            },
            _ => {},
        }

        match criteria.num_max_support {
            Some(x) if used_count >= x => {
                break;
            },
            _ => {},
        }

        let mut idx_best = 0;
        let mut val_best : f64 = 0.0;
        for (e,v) in dict.iter().enumerate() {
            // if used[ e ].is_none() {
                let v = dot( v, &error );
                if v.abs() > val_best.abs() && v.abs() != 0.0 {
                    // println!("v: {:?}, idx best: {}", v, e );
                    val_best = v;
                    idx_best = e;
                }
            // }
        }
        //error if no more progress can be made to reduce error
        if val_best.abs() < 1E-12 {
            break;
        }
        let coeff = magnitude(&error) / val_best;
        // println!("val_best: {}, mag err: {}, coeff:{}", val_best, magnitude(&error), coeff );
        if used[ idx_best ].is_none() {
            used[ idx_best ] = Some( coeff );
        } else {
            *(used[ idx_best ].as_mut().unwrap()) += coeff;
        }
        //update error array: e[i] = e[i] - coeff * arr[i]
        match ( &dict[idx_best], & mut error ) {
            ( Signal::Discrete(x), Signal::Discrete(e) ) => {
                let result = x.iter().zip( e.iter() ).map( |x| x.1 - coeff * x.0 ).collect::<ndarray::Array1<_>>();
                *e = result;
            },
            _ => { panic!("unexpected signal type"); }
        }
        used_count += 1;
    }

    for (e,&v) in used.iter().enumerate(){
        match v {
            Some(x) => {
                sparse.0.push( ( e, x ) ); //found a support with a weighting
            },
            _ => {},
        }
    }

    match criteria.error_threshold {
        Some(x) if magnitude(&error) > x => {
            Err( Error::ErrRemain(magnitude(&error)) )
        },
        _ => {
            Ok( ( sparse, magnitude(&error) ) )
        },
    }
}

#[test]
fn test_matching_pursuit() {
    let dict: Vec< Signal > = vec![ Signal::Discrete( array![ 1., 1., 1., 1. ] ),
                                    Signal::Discrete( array![ 0.5, 1., -1., -0.5 ] ),
                                    Signal::Discrete( array![ -0.5, 1., 1., -0.5 ] ),
                                    Signal::Discrete( array![ 0.5, 1., 0.5, 0. ] ),
                                    Signal::Discrete( array![ 0., 0.5, 1., 0.5 ] ),
                                    Signal::Discrete( array![ 0., 0., 0.2, 1. ] ),
                                    Signal::Discrete( array![ 0.1, 0.2, 0., 0. ] ),
    ];

    let (s,s_len) = {
        let sig = array![ 3.5, 0.5, -0.5, -5.5 ];
        let l = sig.len();
        ( Signal::Discrete( sig ), l )
    };

    let criteria = Criteria {
        error_threshold: None,
        num_max_support: Some(3),
    };
    match matching_pursuit( &dict, &s, &criteria ) {
        Ok( ( Sparse(x), e ) ) => {
            println!("sparse signal: {:#?}, error: {}", x, e );
            let mut reconstruct = vec![0.; s_len ];
            for i in x.iter() {
                let idx = i.0;
                let scale = i.1;
                match &dict[idx] {
                    Signal::Discrete(v) => {
                        for j in 0..v.len() {
                            reconstruct[j] += v[j] * scale;
                        }
                    },
                    _ => {},
                }
            }
            println!("reconstructed: {:#?}", reconstruct );
        },
        Err(e) => {
            match e {
                Error::ErrRemain(r) => {
                    panic!("remaining error: {}", r );
                },
                Error::SupportInvalid => {
                    panic!("invalid support");
                },
                _ => {
                    panic!("unknown error");
                }
            }
        },
    }
}
