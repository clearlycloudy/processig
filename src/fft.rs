extern crate num;

use self::num::complex;
use std::f32;

#[cfg(test)]
use test_common::nearly_equal;

#[cfg(test)]
const E : &f32 = &0.0001;

fn pad_to_nearest_power2( arr: &[f32] ) -> Vec<f32> {
    let n = arr.len();
    let mut i = 1;
    while i < n {
        i = i << 1;
    }
    let num_pad = i - n;
    let mut v = vec![ 0f32; num_pad ];
    v.extend_from_slice( &arr[..] );
    v
}

fn pad_to_nearest_power2_complex( arr: &[complex::Complex<f32>] ) -> Vec<complex::Complex<f32>> {
    let n = arr.len();
    let mut i = 1;
    while i < n {
        i = i << 1;
    }
    let num_pad = i - n;
    let mut v = vec![ complex::Complex::new(0f32,0f32); num_pad ];
    v.extend_from_slice( &arr[..] );
    v
}

pub fn fft_dit( arr: &[f32] ) -> Vec<complex::Complex<f32>> {
    let v = pad_to_nearest_power2( &arr[..] );
    let v_complex = v.iter().map(|x| complex::Complex::new( *x, 0f32 ) ).collect::<Vec<_>>();
    const EXPONENT_SIGN_POS : bool = false;
    cooley_tukey_radix2_dit( &v_complex[..], EXPONENT_SIGN_POS )
}

pub fn fft_dif( arr: &[f32] ) -> Vec<complex::Complex<f32>> {
    let v = pad_to_nearest_power2( &arr[..] );
    let v_complex = v.iter().map(|x| complex::Complex::new( *x, 0f32 ) ).collect::<Vec<_>>();
    const EXPONENT_SIGN_POS : bool = false;
    cooley_tukey_radix2_dif( &v_complex[..], EXPONENT_SIGN_POS )
}

pub fn ifft_dit( arr: &[complex::Complex<f32>] ) -> Vec<f32> {
    let v_complex = pad_to_nearest_power2_complex( &arr[..] );
    const EXPONENT_SIGN_POS : bool = true;
    let v = cooley_tukey_radix2_dit( &v_complex[..], EXPONENT_SIGN_POS );
    let n = v.len();
    let ret = v.iter().map(|x| x.re / n as f32 ).collect::<Vec<f32>>();
    ret
}

pub fn ifft_dif( arr: &[complex::Complex<f32>] ) -> Vec<f32> {
    let v_complex = pad_to_nearest_power2_complex( &arr[..] );
    const EXPONENT_SIGN_POS : bool = true;
    let v = cooley_tukey_radix2_dif( &v_complex[..], EXPONENT_SIGN_POS );
    let n = v.len();
    let ret = v.iter().map(|x| x.re / n as f32 ).collect::<Vec<f32>>();
    ret
}

fn cooley_tukey_radix2_dit( arr: &[complex::Complex<f32>], exponent_sign_pos: bool ) -> Vec<complex::Complex<f32>> {
    //performs radix-2 decimation in time
    //w = e^(sign)*2pi*i/n
    let n = arr.len();
    let mut ret = vec![];
    if 1 == n {
        ret.extend_from_slice( &arr[..] );
        ret
    } else {
        let i = complex::Complex::new( 0f32, 1f32 );
        let exponent = if exponent_sign_pos {
            // complex::Complex::new( 0f32, 2f32 * f32::consts::PI / n as f32 )
            2f32 * f32::consts::PI * i / (n as f32)
        } else {
            // complex::Complex::new( 0f32, -2f32 * f32::consts::PI / n as f32 )
            -2f32 * f32::consts::PI * i / (n as f32)
        };
        let w_base = exponent.exp(); //e^exponent
        let mut w = complex::Complex::new( 1f32, 0f32 );
        let mut y_e = vec![ complex::Complex::new( 0f32, 0f32 ); n/2 ];
        let mut y_o = vec![ complex::Complex::new( 0f32, 0f32 ); n/2 ];
        for j in 0..n/2 {
            y_e[j] = arr[j*2];
            y_o[j] = arr[j*2+1];
        }
        let y_o_ret = cooley_tukey_radix2_dit( &y_o[..], exponent_sign_pos );
        let y_e_ret = cooley_tukey_radix2_dit( &y_e[..], exponent_sign_pos );
        let mut y = vec![ complex::Complex::new( 0f32, 0f32 ); n ];
        for j in 0..n/2 {
            y[j] = y_e_ret[j] + w * y_o_ret[j];
            y[j+n/2] = y_e_ret[j] - w * y_o_ret[j];
            w = w * w_base;
        }
        y
    }
}

fn reverse_bits( mut input: u32, mut num_bits: u32 ) -> u32 {
    let mut ret = 0;
    while num_bits != 0 {
        ret = ( ret << 1 ) | 0b1 & input;
        num_bits -= 1;
        input = input >> 1;
    }
    ret
}

fn cooley_tukey_radix2_dif( arr: &[complex::Complex<f32>], exponent_sign_pos: bool ) -> Vec<complex::Complex<f32>> {
    //iterative DIF 
    let n = arr.len();
    //compute tree level numbers
    let mut levels = 0;
    let mut n_copy = 1;
    while n_copy < n {
        levels += 1;
        n_copy = n_copy << 1;
    }

    //re-arrange bottom level of the tree
    let mut output = vec![ complex::Complex::new( 0f32, 0f32 ); n ];
    for i in 0..n {
        let rev_num = reverse_bits(i as u32, levels);
        output[rev_num as usize] = arr[i];
    }
    for s in 0..levels { //for each tree level
        let m = 1 << (s+1); //2^(s+1) = decimation group size
        let i = complex::Complex::new( 0f32, 1f32 );
        let exponent = if exponent_sign_pos {
            2f32 * f32::consts::PI * i / (m as f32)
        } else {
            -2f32 * f32::consts::PI * i / (m as f32)
        };
        let w_base = exponent.exp(); //e^exponent = primitive m'th root of unity
        let mut k = 0;
        while k < n { //for each butterfly group
            let mut w = complex::Complex::new( 1f32, 0f32 ); //twiddle factor
            for j in 0..m/2 { //for each butterfly pair in current group
                let odd = w * output[ k + j + m/2 ];
                let even = output[ k + j ];
                output[ k + j ] = even + odd;
                output[ k + j + m/2 ] = even - odd;
                w = w * w_base;
            }
            k += m;
        }
    }
    output
}


#[test]
fn test_padding() {
    let arr1 = vec![ 1f32; 16 ];
    let arr2 = vec![ 1f32; 17 ];
    let arr3 = vec![ 1f32; 1 ];
    
    let out1 = pad_to_nearest_power2( &arr1[..] );
    let out2 = pad_to_nearest_power2( &arr2[..] );
    let out3 = pad_to_nearest_power2( &arr3[..] );

    assert_eq!( out1.len(), 16 );
    assert_eq!( out2.len(), 32 );
    assert_eq!( out3.len(), 1 );
}

#[test]
fn test_fft_dit() {
    let arr = vec![ 0., 2., 2., 0. ];
    let out = fft_dit( &arr[..] );
    let expected = vec![ complex::Complex{ re: 4., im: 0. },
                         complex::Complex{ re: -2., im: -2. },
                         complex::Complex{ re: 0., im: 0. },
                         complex::Complex{ re: -2., im: 2. } ];

    assert_eq!( out.len(), 4 );
    expected.iter().zip( out.iter() )
        .for_each( |x| assert!( nearly_equal( &x.0.re, &x.1.re, E ) &&
                                nearly_equal( &x.0.im, &x.1.im, E ) ) );
}

#[test]
fn test_ifft_dit() {

    let arr = vec![ complex::Complex{ re: 4., im: 0. },
                    complex::Complex{ re: -2., im: -2. },
                    complex::Complex{ re: 0., im: 0. },
                    complex::Complex{ re: -2., im: 2. } ];
    let out = ifft_dit( &arr[..] );
    let expected = vec![ 0., 2., 2., 0. ];
    assert_eq!( out.len(), 4 );
    expected.iter().zip( out.iter() )
        .for_each( |x| assert!( nearly_equal( x.0, x.1, E ) &&
                                nearly_equal( x.0, x.1, E ) ) );
}


#[test]
fn test_reverse_bit() {
    let ret = reverse_bits( 0b1110101, 7 );
    assert_eq!( ret, 0b1010111 );
}

#[test]
fn test_fft_dif() {
    let arr = vec![ 0., 2., 2., 0. ];
    let out = fft_dif( &arr[..] );
    let expected = vec![ complex::Complex{ re: 4., im: 0. },
                         complex::Complex{ re: -2., im: -2. },
                         complex::Complex{ re: 0., im: 0. },
                         complex::Complex{ re: -2., im: 2. } ];

    assert_eq!( out.len(), 4 );
    expected.iter().zip( out.iter() )
        .for_each( |x| assert!( nearly_equal( &x.0.re, &x.1.re, E ) &&
                                nearly_equal( &x.0.im, &x.1.im, E ) ) );
}

#[test]
fn test_ifft_dif() {

    let arr = vec![ complex::Complex{ re: 4., im: 0. },
                    complex::Complex{ re: -2., im: -2. },
                    complex::Complex{ re: 0., im: 0. },
                    complex::Complex{ re: -2., im: 2. } ];
    let out = ifft_dif( &arr[..] );
    let expected = vec![ 0., 2., 2., 0. ];
    assert_eq!( out.len(), 4 );
    expected.iter().zip( out.iter() )
        .for_each( |x| assert!( nearly_equal( x.0, x.1, E ) &&
                                nearly_equal( x.0, x.1, E ) ) );
}

#[test]
fn test_fft_dit_2() {
    let arr = vec![ 1., 0., 0., 0., 0., 0., 0., 0. ];
    let out = fft_dit( &arr[..] );
    let expected = vec![ complex::Complex{ re: 1.0, im: 0. }; 8 ];
    
    assert_eq!( out.len(), 8 );
    expected.iter().zip( out.iter() )
        .for_each( |x| assert!( nearly_equal( &x.0.re, &x.1.re, E ) &&
                                nearly_equal( &x.0.im, &x.1.im, E ) ) );
}

#[test]
fn test_fft_dif_2() {
    let arr = vec![ 1., 0., 0., 0., 0., 0., 0., 0. ];
    let out = fft_dif( &arr[..] );
    let expected = vec![ complex::Complex{ re: 1.0, im: 0. }; 8 ];

    assert_eq!( out.len(), 8 );
    expected.iter().zip( out.iter() )
        .for_each( |x| assert!( nearly_equal( &x.0.re, &x.1.re, E ) &&
                                nearly_equal( &x.0.im, &x.1.im, E ) ) );
}

#[test]
fn test_fft_dit_3() {
    let arr = vec![ 0., 1., 0., 0., 0., 0., 0., 0. ];
    let out = fft_dit( &arr[..] );
    let expected = vec![ complex::Complex{ re: 1.,      im: 0. },
                         complex::Complex{ re: 0.7071,  im: -0.7071 },
                         complex::Complex{ re: 0.,      im: -1. },
                         complex::Complex{ re: -0.7071, im: -0.7071 },
                         complex::Complex{ re: -1.,     im: 0. },
                         complex::Complex{ re: -0.7071, im: 0.7071 },
                         complex::Complex{ re: 0.,      im: 1. },
                         complex::Complex{ re: 0.7071,  im: 0.7071 } ];

    assert_eq!( out.len(), 8 );
    expected.iter().zip( out.iter() )
        .for_each( |x| assert!( nearly_equal( &x.0.re, &x.1.re, E ) &&
                                nearly_equal( &x.0.im, &x.1.im, E ) ) );
}

#[test]
fn test_fft_dif_3() {
    let arr = vec![ 0., 1., 0., 0., 0., 0., 0., 0. ];
    let out = fft_dif( &arr[..] );
    let expected = vec![ complex::Complex{ re: 1.,      im: 0. },
                         complex::Complex{ re: 0.7071,  im: -0.7071 },
                         complex::Complex{ re: 0.,      im: -1. },
                         complex::Complex{ re: -0.7071, im: -0.7071 },
                         complex::Complex{ re: -1.,     im: 0. },
                         complex::Complex{ re: -0.7071, im: 0.7071 },
                         complex::Complex{ re: 0.,      im: 1. },
                         complex::Complex{ re: 0.7071,  im: 0.7071 } ];
    assert_eq!( out.len(), 8 );
    expected.iter().zip( out.iter() )
        .for_each( |x| assert!( nearly_equal( &x.0.re, &x.1.re, E ) &&
                                nearly_equal( &x.0.im, &x.1.im, E ) ) );
}
