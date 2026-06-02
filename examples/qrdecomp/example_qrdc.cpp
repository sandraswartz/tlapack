/*
// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>


// <T>LAPACK
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/gemv.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/lapack/unmqr.hpp>


// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

//------------------------------------------------------------------------------
template <typename real_t>
void run(size_t m, size_t n)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = true;


    // Arrays
    std::vector<real_t> tau(n);

    // Matrices
    std::vector<real_t> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<real_t> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<real_t> Q_;
    auto Q = new_matrix(Q_, m, n);
    std::vector<real_t> x_; 
    auto x = new_matrix(x_, n, 1);// solution of the system Ax = b, where A is m*n and x is n*1
    std::vector<real_t> b_;
    auto b = new_matrix(b_, m, 1);



        // Initialize arrays with junk
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                A(i, j) = static_cast<float>(0xDEADBEEF);
            //}
            //for (size_t i = 0; i < n; ++i) {
                //R(i, j) = static_cast<float>(0xFEE1DEAD);
           // }
            tau[j] = static_cast<float>(0xFFBADD11);
        }
 
        A(0, 0) = 1.0;
        A(1, 0) = 2.0;
        A(2, 0) = 2.0;
        A(0, 1) = -4.0;
        A(1, 1) = 3.0;
        A(2, 1) = 2.0;

        //generates a random x matrix
        for (size_t i = 0; i < n; ++i) {
            x(i,0) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        //saves x
        //std::vector<real_t> b(m);
        //for (size_t i = 0; i < m; ++i) {
        //  //      b(i) = x(i);
        //}

        //Calculates Ax = b
        tlapack::gemm(tlapack::Op::NoTrans, 1.0, A, x, 0.0, b);



        //GEQR2
        tlapack::geqr2(A, tau);



        //trtrs, tlapack::Op::NoTrans
    ///for (size_t i = 4; i >= 1; --i){
       // x(i) = b(i)
        //for (j = i + 1:4){
         //   x(i) = x(i) - r(i,j)*x(j)

        //}
        //x(i) = x(i)/r(i, i)

    //}


}
}


int main(int argc, char** argv){

    int m = 3;
    int n = 2;

    run<float>(m, n);
    
    







}

*/

/// @file example_geqrf_solve.cpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/lapack/unmqr.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Arrays
    std::vector<T> tau(n);

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_orig_;
    auto A_orig = new_matrix(A_orig_, m, n);
    std::vector<T> x_;
    auto x = new_matrix(x_, n, 1);
    std::vector<T> b_;
    auto b = new_matrix(b_, m, 1);
    std::vector<T> b_orig_;
    auto b_orig = new_matrix(b_orig_, m, 1);

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            A(i, j) = static_cast<T>(0xDEADBEEF);
            A_orig(i, j) = static_cast<T>(0xCAFED00D);
        }
        tau[j] = static_cast<T>(0xFFBADD11);
        x(j, 0) = static_cast<T>(0xBAADF00D);
        b(j, 0) = static_cast<T>(0xBAADF00D);
    }

    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i)
            A(i, j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        x(j, 0) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }

    // Copy A to A_orig
    tlapack::lacpy(tlapack::GENERAL, A, A_orig);

    // Copy b to b_orig
    tlapack::lacpy(tlapack::GENERAL, b, b_orig);

    // Generate b = A*x so that the solution is known
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans,
                  static_cast<T>(1.0), A, x, static_cast<T>(0.0), b);

    // b_0_n receives the first n rows of b (this is the part of b that is used
    // in the solution)
    auto b_0_n = slice(b, range(0, n), range(0, 1));

    // R is a slice of A that will receive the upper triangular factor R from
    // the QR factorization
    auto R = slice(A, range(0, n), range(0, n));

    // QR factorization
    tlapack::geqrf(A, tau);

    // Apply Q' to b
    tlapack::unmqr(tlapack::Side::Left, tlapack::Op::ConjTrans, A, tau, b);

    // Solve R*x = Q'*b
    tlapack::trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                  tlapack::Op::NoTrans, tlapack::Diag::NonUnit,
                  static_cast<T>(1.0), R, b_0_n);

    // Print A, b, and x
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(A_orig);
        std::cout << std::endl << "b = ";
        printMatrix(b_orig);
        std::cout << std::endl << "x = ";
        printMatrix(b_0_n);
    }

    // Compute ||b - A*x||_F / ||A||_F
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans,
                  static_cast<T>(-1.0), A_orig, b_0_n, static_cast<T>(1.0),
                  b_orig);

    // Frobenius norm of A
    auto normA = tlapack::lange(tlapack::FROB_NORM, A_orig);
    auto norm_residual = tlapack::lange(tlapack::FROB_NORM, b_orig) / normA;

    // Output
    std::cout << std::endl;
    std::cout << "||b - A*x||_F/||A||_F = " << norm_residual;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n;

    // Default arguments
    m = (argc < 2) ? 7 : atoi(argv[1]);
    n = (argc < 3) ? 5 : atoi(argv[2]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", m, n);
    run<float>(m, n);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", m, n);
    run<double>(m, n);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )", m, n);
    run<long double>(m, n);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )", m, n);
    run<std::complex<float>>(m, n);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d )", m, n);
    run<std::complex<double>>(m, n);
    printf("-----------------------\n");

    printf("run< complex<long double> >( %d, %d )", m, n);
    run<std::complex<long double>>(m, n);
    printf("-----------------------\n");

    return 0;
}
