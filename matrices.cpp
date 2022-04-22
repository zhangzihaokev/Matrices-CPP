//
//  main.cpp
//  matrix class
//
//  Created by Kevin Zhang on 12/11/18.
//  Copyright © 2018 Kevin Zhang. All rights reserved.
//

#include <iostream>
#include <math.h>
#include <random>
#include <iterator>
#include <array>


using namespace std;


int sgn(double x){
    return (x < 0) ? -1 : 1;
}

//constructs m x n matrix with random integer valued entries from 0 - 10
//row-major storage
double * construct_matrix(int m, int n){
    double * A = new double[m*n];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            A[j+i*n] = rand() % 10;
        }
    }
    return A;
}

double * construct_uptri(int m){
    double * A = new double[m*m];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < m; ++j){
            A[j+i*m] = (i<=j) ? rand() % 10 : 0.0;
        }
    }
    return A;
}

double * construct_lowtri(int m){
    double * A = new double[m*m];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < m; ++j){
            A[j+i*m] = (i>=j) ? rand() % 10 : 0.0;
        }
    }
    return A;
}


double * construct_zeros(int m, int n){
    double * Z = new double[m*n];
    fill_n(Z,m*n,0.0);
    return Z;
}

//prints matrix A
void print_matrix(const double * A, int rows, int cols){
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            cout << A[j+i*cols] << " ";
        }
        cout << endl;
    }
}

//matrix addition, adds B to A (modifies A)
double * add_matrix(double * A, const double * B, int rows, int cols){
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            A[j+i*cols] +=  B[j+i*cols];
        }
    }
    return A;
}


//scalar multiply matrix A by integer val, computes val*A
double * scalar_mult(double * A, int rows, int cols, double val){
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            A[j+i*cols] = val*A[j+i*cols];
        }
    }
    return A;
}

//returns the result of naive matrix matrix multiplication; A is m x l, B is l x n, C is m x n
double *  dumb_multiply(const double * A, const double * B, int m, int l, int n){
    double * C = new double[m*n];
    fill_n(C,m*n,0.0);
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            double sum = 0.0;
            for (int k = 0; k < l; ++k){
                sum += A[k+i*l]*B[j+k*n];
            }
            C[j+i*n] = sum;
        }
    }
    return C;
}


//returns the result of blocked matix matrix multiplication; A is m x l, B is l x n, C is m x n
double * multiply(const double * A, const double * B, int m, int l, int n, int b){
    double * C = new double[m*n];
    fill_n(C,m*n,0.0);
    for (int I = 0; I < m; I += b){
        for (int J = 0; J < n; J += b){
            for (int K = 0; K < l; K += b){
                for (int i = I; i < min(I+b,m); ++i){
                    for (int j = J; j < min(J+b,n); ++j){
                        double sum = 0.0;
                        for (int k = K; k < min(K+b,l); ++k){
                            sum += A[k+i*l]*B[j+k*n];
                        }
                        C[j+i*n] += sum;
                    }
                }
            }
        }
    }
    return C;
}


//returns the transpose of B = A^T
double * transpose(const double * A, int rows, int cols){
    double * B = new double[rows*cols];
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            B[j+i*cols] = A[i+j*rows]; //switches from col-major to row-major
        }
    }
    return B;
}

//creates an rows x cols identity matrix
double *  identity(int rows, int cols){
    double * I = new double[rows*cols];
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            I[j+i*cols] = (i == j) ? 1 : 0;
        }
    }
    return I;
}

//returns the (b-a+1) x (d-c+1) minor matrix of A, [a,b] is row range, [c,d] is col range
double * minor(double * A, int m, int n, int a, int b, int c, int d){
    double * B = new double[(b-a+1)*(d-c+1)];
    for (int i = 0; i < (b-a+1); ++i){
        for (int j = 0; j < (d-c+1); ++j){
            B[j+i*(d-c+1)] = A[(j+c)+(i+a)*n];
        }
    }
    return B;
}

//returns ij element of matrix A
double get_ij(const double * A, int i, int j, int cols){
    return A[j+i*cols];
}

//sets ij element to val
void set_ij(double * A, int i, int j, int cols, double val){
    A[j+i*cols] = val;
}

//returns the ith row of matrix A
double * extract_row(const double * A, int rows, int cols, int i){
    double * v = new double[cols];
    for (int j = 0; j < cols; ++j){
        v[j] = A[j+i*cols];
    }
    return v;
}

//returns the jth column of matrix A
double * extract_col(double * A, int rows, int cols, int j){
    double * v = new double[rows];
    for (int i = 0; i < rows; ++i){
        v[i] = A[j+i*cols];
    }
    return v;
}

//returns the dot product of vector a and b
double dot(double * a, double * b, int dim){
    double result = 0.0;
    for (int i = 0; i < dim; ++i){
        result += a[i]*b[i];
    }
    return result;
}

//returns 2 norm of vector a
double norm(double * a, int dim){
    double result = pow(dot(a,a,dim),0.5);
    return result;
}

//computes the QR decomposition of A via modified Gram Schmidt
void QR(double * A, double * Q, double * R, int rows, int cols){
    double * v = new double[rows*cols];
    double * q = new double[rows];
    fill_n(Q,rows*cols,0.0);
    fill_n(R,rows*cols,0.0);
    
    memcpy(v, A, rows*cols*sizeof(double)); //copy A into v
    for (int i = 0; i < cols; ++i){
        double * v_i = extract_col(v, rows, cols, i); //ith column of v
        R[i+i*cols] = norm(v_i, rows); //r_ii = ||v_i||_2
        memcpy(q, v_i, rows*sizeof(double)); //q_i = v_i
        scalar_mult(q, rows, 1, 1/norm(v_i, rows)); //q_i/r_ii

        //build Q col by col, building the ith column here
        for (int k = 0; k < rows; ++k){
            Q[i+k*cols] = q[k];
        }
        
        delete [] v_i;
        for (int j = i+1; j < cols ; ++j){
            double * q_i = new double[rows];
            memcpy(q_i,q,rows*sizeof(double)); //this is necessary so that the columns of Q are not changed within the loop
            double * v_j = extract_col(v, rows, cols, j); //jth column of v
            R[j+i*cols] = dot(q_i,v_j,rows); //r_ij = q_i * v_j
            scalar_mult(q_i, rows, 1, -1*R[j+i*cols]); //-r_ij*q_i
            add_matrix(v_j, q_i, rows, 1); // v_j - r_ij*q_i
            for (int k = 0; k < rows; ++k){ //update the next column of v
                v[j+k*cols] = v_j[k];
            }
            delete [] q_i;
            delete [] v_j;
        }
    }
    
    cout << "Q = " << endl;
    print_matrix(Q, rows, cols);
    cout<<"\n"<<endl;
    cout << "R = " << endl;
    print_matrix(R, rows, cols);
    cout << "\n" << endl;
    
    
    
    double * q_0 = extract_col(Q, rows, cols, 0);
    double * q_1 = extract_col(Q, rows, cols, 1);
    cout << "are the columns orthogonal?"<<endl;
    cout << dot(q_0,q_1,rows) << endl;
    cout << "\n" << endl;
    
    cout << "are the columns normal?"<<endl;
    cout << dot(q_0,q_0,rows) << endl;
    cout << "\n" << endl;
    

    //de-allocate
    delete [] v;
    delete [] q;

    delete [] q_0;
    delete [] q_1;
}

/*
//reduces A into its upper triangular form via Householder transforms
void house(double * A, int rows, int cols){
    for (int k = 0; k < cols; ++k){
        double * x = minor(A, rows, cols, k, rows, k, k);
        double x_1 = get_ij(x, 0, 0, rows-k+1);
        double * e_1 = identity(rows-k+1, 1);
        scalar_mult(e_1, rows-k+1, 1, sgn(x_1)*norm(x, rows-k+1));
        add_matrix(x, e_1, rows-k+1, 1);
        scalar_mult(x, rows-k+1, 1, 1/norm(x, rows-k+1));
        
        
        delete [] x;
    }
    
}
*/
//returns the Cholesky factor L of a SPD matrix A
double * cholesky(double * A, int n){
    double * L = new double[n*n];
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < i+1; ++j){
            double s = 0.0;
            for (int k = 0; k < j; ++k){
                s += L[k+i*n]*L[k+j*n];
            }
            L[j+i*n] = (i == j) ? sqrt(A[i+i*n]-s) : (1.0/L[j+j*n]*(A[j+i*n]-s));
        }
    }
    return L;
}

//computes the LU decomposition of n x n matrix A
void LU(double * A, int n){
    double * L = construct_zeros(n, n);
    double * U = construct_zeros(n, n);
    
    for (int i = 0; i < n; ++i){
        for (int k = i; k < n; ++k){
            double sum = 0.0;
            for (int j = 0; j < n; ++j){
                sum += L[j+i*n]*U[k+j*n];
            }
            U[k+i*n] = A[k+i*n] - sum;
        }
        for (int k = i; k < n; ++k){
            if (i == k){
                L[i+i*n] = 1.0;
            }
            else{
                double sum = 0.0;
                for (int j = 0; j < n; ++j){
                    sum += L[j+k*n]*U[i+j*n];
                }
                L[i+k*n] = (A[i+k*n] - sum)/U[i+i*n];
            }
        }
    }
    /*
    cout << "lower triangular factor L = " << endl;
    print_matrix(L, n, n);
    cout << "\n";
    cout << "does it work?" << endl;
    print_matrix(U, n, n);
    cout << "\n";
    double * result = multiply(L, U, 3, 3, 3, 1);
    cout << "test LU = " << endl;
    print_matrix(result, 3, 3);
    
    delete [] L;
    delete [] U;
     */
}
//solves Ax=b where A is upper triangular and nonsingular
double * back_substitution(double * A, double * b, int n){
    double * x = new double[n];
    fill_n(x,n,0.0);
    
    for (int i = n; i >= 0; i--){
        double s = 0.0;
        for (int j = i+1; j < n; ++j){
            s += A[j+i*n]*x[j];
        }
        x[i] = (b[i] - s)/A[i+i*n];
    }
    return x;
}
//solves Ax=b where A is lower triangular and nonsingular
double * forward_substitution(double * A, double * b, int n){
    double * x = new double[n];
    for (int i = 0; i < n; ++i){
        double s = 0;
        for (int j = 0; j < i; ++j){
            s += A[j+i*n]*x[j];
        }
        x[i] = (b[i] - s)/A[i+i*n];
    }
    return x;
}

int main() {
    
    
    return 0;
}

