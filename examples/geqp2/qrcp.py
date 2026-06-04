import numpy as np

def printMatrix(A):
    m, n = A.shape
    for i in range(0,m):
        for j in range(0,n):
            print("{:13.5e}".format(A[i,j]), end="")
        print("\n", end="")


def larfg(alpha, x):
    x_norm = np.linalg.norm(x, 2)
    if x_norm == 0.0:
        return alpha, x.copy(), 0.0
    beta = -np.sign(alpha) * np.hypot(alpha, x_norm)
    if beta == 0.0:
        beta = -x_norm
    tau = (beta - alpha) / beta
    x_new = x / (alpha - beta)
    return beta, x_new, tau


def larf(v, tau, C):
    v = np.array([1.0, *v])
    v[0] = 1.0
    v_col = v.reshape(-1, 1)
    C -= tau * (v_col @ (v_col.T @ C))
    return C


def orgqr(V, tau, C):
    m, n = C.shape
    for j in range(n-1,-1,-1):
        C[j:m,j:] = larf(V[j+1:m,j], tau[j], C[j:m,j:])
    return C


def qr_column_pivoting(A, tol=None):
    m, n = A.shape

    Q = np.zeros((m,n))
    for j in range(n):
        Q[j, j] = 1.0
    R = np.zeros((n,n))
    tau = np.zeros(n)
    pivots = np.arange(n)

    # Compute all of the initial column norms squared
    s = np.zeros(n)
    for j in range(n):
        s[j] = np.sum(A[:, j] ** 2)
    print("Initial column norms squared: ", s)

    for k in range(n):
        # Identify the pivot column (maximum remaining norm)
        # pivot_idx = k + np.argmax(s[k:])
        max_norm = s[k];
        pivot_idx = k;
        # for j in range(k+1,n):
        #     if (s[j] > max_norm):
        #         max_norm = s[j];
        #         pivot_idx = j;
            
        print(f"Step {k}: Pivot column index = {pivot_idx}, norm squared = {s[pivot_idx]:.2e}")

        # Swap columns globally across A, pivots array, and tracking vectors
        if pivot_idx != k:
            A[:, [k, pivot_idx]] = A[:, [pivot_idx, k]]
            pivots[[k, pivot_idx]] = pivots[[pivot_idx, k]]
            s[[k, pivot_idx]] = s[[pivot_idx, k]]

        # Compute the Householder reflector to zero out under diagonal elements
        A[k,k], A[k+1:,k], tau[k] = larfg(A[k,k], A[k+1:m,k])

        # Apply reflector to the trailing matrix A
        A[k:m,k+1:] = larf(A[k+1:m,k], tau[k], A[k:m,k+1:])

        # downdate the norms for the remaining columns
        for j in range(k,n):
            s[j] -= A[k,j]*A[k,j]

    # apply Householder reflectors to Q
    Q = orgqr(A, tau, Q)

    # copy upper triangular of A into R
    for j in range(n):
        for i in range(j+1):
            R[i,j] = A[i,j]

    P = np.eye(n)[:, pivots]
    return Q, R, P


def qr_column_pivoting_stewart(A):
    m, n = A.shape
    Q = np.zeros((m,n))
    for j in range(n):
        Q[j, j] = 1.0
    tau = np.zeros(n)
    pivots = np.arange(n)

    # 1. Initialize Stewart's dual-vector tracking tracking structures
    gamma = np.zeros(n)       # Current dynamic norm estimate
    gamma_init = np.zeros(n)  # Baseline initial exact norm
    for j in range(n):
        gamma[j] = np.linalg.norm(A[:, j], 2)
        gamma_init[j] = gamma[j]

    # Standard LAPACK guard thresholds
    eps = np.finfo(float).eps
    tau1 = np.sqrt(eps)  # Primary guard (relative decay against initial max)
    tau2 = 0.05          # Secondary guard (local sharp drop)

    for k in range(min(m, n)):
        # 2. Identify the pivot column (maximum remaining norm)
        pivot_idx = k + np.argmax(gamma[k:])

        # Swap columns globally across R, pivots array, and tracking vectors
        if pivot_idx != k:
            A[:, [k, pivot_idx]] = A[:, [pivot_idx, k]]
            tau[[k, pivot_idx]] = tau[[pivot_idx, k]]
            pivots[[k, pivot_idx]] = pivots[[pivot_idx, k]]
            gamma[[k, pivot_idx]] = gamma[[pivot_idx, k]]
            gamma_init[[k, pivot_idx]] = gamma_init[[pivot_idx, k]]

        # 3. Compute the Householder reflector to zero out under diagonal elements
        A[k,k], A[k+1:,k], tau[k] = larfg(A[k,k], A[k+1:m,k])

        # Apply reflector matrix to the trailing matrix A
        A[k:m,k+1:] = larf(A[k+1:m,k], tau[k], A[k:m,k+1:])

        # 4. Process Stewart's Dual-Vector Tracking Guard for trailing columns
        for j in range(k + 1, n):
            if gamma[j] > 0:
                r_kj = A[k, j]

                # Check for numerical boundaries before calculating square roots
                ratio = abs(r_kj) / gamma[j]
                if ratio < 1.0:
                    gamma_new = gamma[j] * np.sqrt(1.0 - ratio**2)
                else:
                    gamma_new = 0.0

                # TEST 1: Absolute / Historical Decay Guard
                if (gamma_new / gamma_init[j]) <= tau1:
                    # Catastrophic drift detected: enforce full O(m-k) column re-computation
                    gamma[j] = np.linalg.norm(A[k+1:m, j], 2)
                    gamma_init[j] = gamma[j]  # Reset historical baseline anchor

                # TEST 2: Local Step Perturbation Guard
                elif (gamma_new / gamma[j]) <= tau2:
                    # Steep geometric collapse over a single step: compute local slice
                    gamma[j] = np.linalg.norm(A[k+1:m, j], 2)

                else:
                    # The cheap O(1) mathematical downdate expression is deemed safe
                    gamma[j] = gamma_new

    # apply Householder reflectors to Q
    Q = orgqr(A, tau, Q)

    # copy upper triangular of A into R
    R = np.zeros((n,n))
    for j in range(n):
        for i in range(j+1):
            R[i,j] = A[i,j]

    # Construct the explicit permutation matrix mapping input indexes
    P = np.eye(n)[:, pivots]
    return Q, R, P

# ==========================================
# Verification & Numerical Performance Test
# ==========================================
if __name__ == "__main__":

    # import matplotlib.pyplot as plt

    # Initialize the generator with a specific seed
    rng = np.random.default_rng(seed=42)

    m = 7

    # Create a Kahn Matrix (ill conditioned)
    C = np.eye(m)
    S = np.eye(m)
    c = np.cos(np.pi /12)
    s = np.sin(np.pi / 12)
    # c = 0.4664999999999993
    # s = np.sqrt(1 - c**2)
    for j in range(1, m):
        S[j,j] = s**j
        for i in range(0, j):
            C[i,j] = -1.0 * c

    print("S = ")
    printMatrix(S)
    print("C = ")
    printMatrix(C)
    # r = 3
    # S = rng.random((m,r))
    # C = rng.random((r,m))
    A_orig = S @ C

    # Scramble the order of the columns so that the algorithm is forced to pivot
    # scramble_order = rng.permutation(m)
    # print(scramble_order)
    # scramble_order = [0, 2, 1, 4, 3, 6, 5]

    # A_orig = A_orig[:, scramble_order]

    # Run Businger and Golub's QR Factorization
    A = A_orig.copy()
    print("A = ")
    printMatrix(A)
    Q_bg, R_bg, P_bg = qr_column_pivoting(A)
    print("Q = ")
    printMatrix(Q_bg)
    print("R = ")
    printMatrix(R_bg)
    print("P = ")
    printMatrix(P_bg)

    AA = A_orig.copy()
    Q, R = np.linalg.qr(AA)
    print("Q (numpy) = ")
    printMatrix(Q)
    print("R (numpy) = ")
    printMatrix(R)

    # Check algebraic constraints
    reconstruction_err = np.linalg.norm(A_orig @ P_bg - Q_bg @ R_bg)
    orthogonality_err = np.linalg.norm(Q_bg.T @ Q_bg - np.eye(Q_bg.shape[0]))

    # d = int(np.log10(m) // 1 + 1)
    # my_string = "  R[{{:{:}}},{{:{:}}}]: {{:12.4e}}".format(d,d)

    print("--- Businger and Golub's QR Factorization Execution ---")
    print(f"Reconstruction Accuracy ||AP - QR||: {reconstruction_err:.2e}")
    print(f"Orthogonality Retention ||Q^T*Q - I||: {orthogonality_err:.2e}")
    # print("\nResulting Diagonal Elements of R (Rank-revealing sorting order):")
    # for idx, r_diag in enumerate(np.diag(R)):
        # print(my_string.format(idx,idx,r_diag))

    print("\n\n")

    # Run Stewart's QR CP Factorization
    A = A_orig.copy()
    Q_s, R_s, P_s = qr_column_pivoting_stewart(A)
    print("Q = ")
    printMatrix(Q_s)
    print("R = ")
    printMatrix(R_s)
    print("P = ")
    printMatrix(P_s)

    # Check algebraic constraints
    reconstruction_err = np.linalg.norm(A_orig @ P_s - Q_s @ R_s)
    orthogonality_err = np.linalg.norm(Q_s.T @ Q_s - np.eye(Q_s.shape[0]))

    print("--- Stewart's QR Factorization Execution ---")
    print(f"Reconstruction Accuracy ||AP - QR||: {reconstruction_err:.2e}")
    print(f"Orthogonality Retention ||Q^T*Q - I||: {orthogonality_err:.2e}")
    # print("\nResulting Diagonal Elements of R (Rank-revealing sorting order):")
    # for idx, r_diag in enumerate(np.diag(R)):
        # print(my_string.format(idx,idx,r_diag))

    """
    plt.figure()
    # Create a float copy to allow NaN values
    matrix_float_bg = R_bg.astype(float)
    matrix_float_s = R_s.astype(float)
    # Mask the diagonal and lower triangle with True
    mask_bg = np.tril(np.ones(R_bg.shape), k=0).astype(bool)
    mask_s = np.tril(np.ones(R_s.shape), k=0).astype(bool)
    # Apply absolute values and set masked positions to NaN
    abs_matrix_bg = np.abs(matrix_float_bg)
    abs_matrix_bg[mask_bg] = np.nan
    abs_matrix_s = np.abs(matrix_float_s)
    abs_matrix_s[mask_s] = np.nan
    # Compute max ignoring NaNs
    max_values_bg = np.nanmax(abs_matrix_bg, axis=1)
    max_values_s = np.nanmax(abs_matrix_s, axis=1)
    # Compute max ignoring NaNs
    plt.semilogy(list(range(m-2)), np.abs(np.diag(R_bg))[:-2], color = 'blue', linestyle='-', label='$|R_{ii}|$ for Businger and Golub')
    plt.semilogy(list(range(m-2)), max_values_bg[:-2], color = 'blue', linestyle=':', label='$\mu_i$ for Businger and Golub')
    plt.semilogy(list(range(m-2)), np.abs(np.diag(R_s))[:-2], color = 'green', linestyle='-', label='$|R_{ii}|$ for Stewart')
    plt.semilogy(list(range(m-2)), max_values_s[:-2], colorP = 
  0.00000e+00  0.00000e+00  0.00000e+00  1.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00
  0.00000e+00  0.00000e+00  1.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00
  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  1.00000e+00
  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  1.00000e+00  0.00000e+00  0.00000e+00
  0.00000e+00  1.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00
  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  1.00000e+00  0.00000e+00
  1.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00
    plt.legend()
    plt.savefig("QRCP_comparison_plot.pdf", bbox_inches="tight")
    plt.show()
    """
